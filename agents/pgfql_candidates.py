import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ConditionalActorVectorField, NextStateVectorField, Value


class PGFQLCandidatesAgent(flax.struct.PyTreeNode):
    """PGFQL with value-guided successor-state candidate selection.

    Training keeps the original PGFQL state-flow, action-distillation, actor-Q,
    and critic objectives.  An auxiliary IQL-style expectile value network is
    learned from dataset state-action pairs.  At inference, the state flow
    proposes multiple successor states, the value network selects one, and the
    original one-step action network generates a single action conditioned on
    the selected state.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(advantage, expectile):
        """Return the elementwise asymmetric squared expectile loss."""
        weight = jnp.where(advantage >= 0, expectile, 1 - expectile)
        return weight * jnp.square(advantage)

    def value_loss(self, batch, grad_params):
        """Fit V(s) to an expectile of the target critics on dataset actions."""
        target_qs = self.network.select('target_critic')(
            batch['observations'],
            actions=batch['actions'],
        )
        target_q = jax.lax.stop_gradient(target_qs.min(axis=0))
        value = self.network.select('value')(
            batch['observations'],
            params=grad_params,
        )
        advantage = target_q - value
        value_loss = self.expectile_loss(
            advantage,
            self.config['expectile'],
        ).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': value.mean(),
            'v_max': value.max(),
            'v_min': value.min(),
            'target_q_mean': target_q.mean(),
            'adv_mean': advantage.mean(),
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the original PGFQL critic loss with one next action."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self._sample_single_action(
            batch['next_observations'],
            seed=sample_rng,
        )
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(
            batch['next_observations'],
            actions=next_actions,
        )
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        q = self.network.select('critic')(
            batch['observations'],
            actions=batch['actions'],
            params=grad_params,
        )
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the original PGFQL state-flow and actor objectives."""
        batch_size, action_dim = batch['actions'].shape
        _, observation_dim = batch['observations'].shape
        rng, state_noise_rng, time_rng = jax.random.split(rng, 3)

        # Original conditional state-flow matching loss.
        state_noise = jax.random.normal(
            state_noise_rng,
            (batch_size, observation_dim),
        )
        target_next_observations = batch['next_observations']
        times = jax.random.uniform(time_rng, (batch_size, 1))
        interpolated_states = (
            (1 - times) * state_noise + times * target_next_observations
        )
        target_velocity = target_next_observations - state_noise
        predicted_velocity = self.network.select('state_stitch')(
            batch['observations'],
            interpolated_states,
            times,
            params=grad_params,
        )
        state_flow_loss = jnp.mean(
            jnp.square(predicted_velocity - target_velocity)
        )

        # Original PGFQL action distillation uses a flow-generated successor.
        rng, next_state_rng = jax.random.split(rng)
        next_state_noises = jax.random.normal(
            next_state_rng,
            (batch_size, observation_dim),
        )
        generated_next_observations = self.compute_flow_next_state(
            batch['observations'],
            noises=next_state_noises,
        )

        rng, action_noise_rng = jax.random.split(rng)
        action_noises = jax.random.normal(
            action_noise_rng,
            (batch_size, action_dim),
        )
        actor_actions = self.network.select('actor_onestep_flow')(
            batch['observations'],
            action_noises,
            generated_next_observations,
            params=grad_params,
        )
        distill_loss = jnp.mean(jnp.square(actor_actions - batch['actions']))

        # Original PGFQL directly optimizes the actor through the critic.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(
            batch['observations'],
            actions=actor_actions,
        )
        q = jnp.mean(qs, axis=0)
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            scale = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = scale * q_loss

        actor_loss = (
            state_flow_loss
            + self.config['alpha'] * distill_loss
            + q_loss
        )

        # Keep the original single-sample action MSE logging path.  Candidate
        # selection must not silently enlarge the training computation graph.
        logged_actions = self._sample_single_action(
            batch['observations'],
            seed=rng,
        )
        mse = jnp.mean(jnp.square(logged_actions - batch['actions']))

        return actor_loss, {
            'actor_loss': actor_loss,
            'next_ob_flow_loss': state_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute original PGFQL losses plus the auxiliary value loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for key, value in value_info.items():
            info[f'value/{key}'] = value

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(
            batch,
            grad_params,
            critic_rng,
        )
        for key, value in critic_info.items():
            info[f'critic/{key}'] = value

        actor_loss, actor_info = self.actor_loss(
            batch,
            grad_params,
            actor_rng,
        )
        for key, value in actor_info.items():
            info[f'actor/{key}'] = value

        total_loss = (
            critic_loss
            + actor_loss
            + self.config['value_loss_weight'] * value_loss
        )
        return total_loss, info

    def target_update(self, network, module_name):
        """Soft-update a target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda parameter, target_parameter: (
                parameter * self.config['tau']
                + target_parameter * (1 - self.config['tau'])
            ),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update PGFQL and the auxiliary expectile value network."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    def _sample_single_action(self, observations, seed):
        """Sample one action exactly as in the original PGFQL."""
        next_state_seed, _ = jax.random.split(seed)
        observation_ndim = len(self.config['ob_dims'])
        batch_shape = observations.shape[:-observation_ndim]

        next_state_noises = jax.random.normal(
            next_state_seed,
            (*batch_shape, *self.config['ob_dims']),
        )
        generated_next_observations = self.compute_flow_next_state(
            observations,
            noises=next_state_noises,
        )

        action_seed, _ = jax.random.split(next_state_seed)
        action_noises = jax.random.normal(
            action_seed,
            (*batch_shape, self.config['action_dim']),
        )
        actions = self.network.select('actor_onestep_flow')(
            observations,
            action_noises,
            generated_next_observations,
        )
        return jnp.clip(actions, -1, 1)

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Select a successor with V and generate one PGFQL action from it."""
        del temperature
        next_state_seed, _ = jax.random.split(seed)
        observation_ndim = len(self.config['ob_dims'])
        batch_shape = observations.shape[:-observation_ndim]
        candidate_axis = len(batch_shape)
        candidate_shape = (
            *batch_shape,
            self.config['num_candidates'],
            *self.config['ob_dims'],
        )

        candidate_observations = jnp.broadcast_to(
            jnp.expand_dims(observations, axis=candidate_axis),
            candidate_shape,
        )
        state_noises = jax.random.normal(next_state_seed, candidate_shape)
        candidate_next_observations = self.compute_flow_next_state(
            candidate_observations,
            noises=state_noises,
        )

        candidate_values = self.network.select('value')(
            candidate_next_observations,
        )
        best_indices = jnp.argmax(candidate_values, axis=-1)

        index_shape = (
            *batch_shape,
            1,
            *([1] * observation_ndim),
        )
        gather_shape = (
            *batch_shape,
            1,
            *self.config['ob_dims'],
        )
        gather_indices = jnp.broadcast_to(
            jnp.reshape(best_indices, index_shape),
            gather_shape,
        )
        selected_next_observations = jnp.take_along_axis(
            candidate_next_observations,
            gather_indices,
            axis=candidate_axis,
        )
        selected_next_observations = jnp.squeeze(
            selected_next_observations,
            axis=candidate_axis,
        )

        action_seed, _ = jax.random.split(next_state_seed)
        action_noises = jax.random.normal(
            action_seed,
            (*batch_shape, self.config['action_dim']),
        )
        actions = self.network.select('actor_onestep_flow')(
            observations,
            action_noises,
            selected_next_observations,
        )
        return jnp.clip(actions, -1, 1)

    @jax.jit
    def compute_flow_next_state(self, observations, noises):
        """Sample successor states from the original PGFQL Euler flow."""
        if self.config['encoder'] is not None:
            observations = self.network.select('state_stitch_encoder')(
                observations
            )

        next_state = noises
        for step in range(self.config['flow_steps']):
            times = jnp.full(
                (*observations.shape[:-1], 1),
                step / self.config['flow_steps'],
            )
            velocities = self.network.select('state_stitch')(
                observations,
                next_state,
                times,
                is_encoded=True,
            )
            next_state = next_state + velocities / self.config['flow_steps']
        return next_state

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create PGFQL plus one IQL-style state-value network."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['state_stitch'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('value'),
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_onestep_flow_def = ConditionalActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )
        state_stitch_def = NextStateVectorField(
            hidden_dims=config['actor_hidden_dims'],
            state_dim=ob_dims,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('state_stitch'),
        )

        network_info = {
            'value': (value_def, (ex_observations,)),
            'critic': (critic_def, (ex_observations, ex_actions)),
            'target_critic': (
                copy.deepcopy(critic_def),
                (ex_observations, ex_actions),
            ),
            'actor_onestep_flow': (
                actor_onestep_flow_def,
                (ex_observations, ex_actions, ex_observations),
            ),
            'state_stitch': (
                state_stitch_def,
                (ex_observations, ex_observations, ex_times),
            ),
        }
        if encoders.get('state_stitch') is not None:
            network_info['state_stitch_encoder'] = (
                encoders['state_stitch'],
                (ex_observations,),
            )

        networks = {key: value[0] for key, value in network_info.items()}
        network_args = {key: value[1] for key, value in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(
            network_def,
            network_params,
            tx=network_tx,
        )

        network.params['modules_target_critic'] = network.params['modules_critic']
        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='pgfql_candidates',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            alpha=10.0,
            expectile=0.7,
            value_loss_weight=1.0,
            num_candidates=16,
            flow_steps=10,
            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config


# Register through config-module import so main.py needs no changes.
from agents import agents as _agent_registry

_agent_registry['pgfql_candidates'] = PGFQLCandidatesAgent
