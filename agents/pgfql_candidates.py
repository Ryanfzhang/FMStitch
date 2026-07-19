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
    """PGFQL with batch-vectorized state-action candidate selection.

    For each state, the state flow proposes K successor states in parallel and
    the original one-step actor produces K corresponding actions.  The online
    double critic selects the candidate with the largest conservative Q score.
    The same candidate policy is used by actor training, critic targets, and
    environment interaction.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _generate_candidates(self, observations, seed, actor_params=None):
        """Generate and score K candidates with a single batched network call."""
        next_state_seed, action_seed = jax.random.split(seed)
        observation_ndim = len(self.config['ob_dims'])
        batch_shape = observations.shape[:-observation_ndim]
        candidate_axis = len(batch_shape)
        candidate_observation_shape = (
            *batch_shape,
            self.config['num_candidates'],
            *self.config['ob_dims'],
        )

        candidate_observations = jnp.broadcast_to(
            jnp.expand_dims(observations, axis=candidate_axis),
            candidate_observation_shape,
        )

        state_noises = jax.random.normal(
            next_state_seed,
            candidate_observation_shape,
        )
        candidate_next_observations = self.compute_flow_next_state(
            candidate_observations,
            noises=state_noises,
        )

        action_noises = jax.random.normal(
            action_seed,
            (
                *batch_shape,
                self.config['num_candidates'],
                self.config['action_dim'],
            ),
        )
        raw_candidate_actions = self.network.select('actor_onestep_flow')(
            candidate_observations,
            action_noises,
            candidate_next_observations,
            params=actor_params,
        )
        candidate_actions = jnp.clip(raw_candidate_actions, -1, 1)

        candidate_qs = self.network.select('critic')(
            candidate_observations,
            actions=candidate_actions,
        )
        candidate_scores = candidate_qs.min(axis=0)
        best_indices = jnp.argmax(candidate_scores, axis=-1)

        gather_indices = jnp.broadcast_to(
            best_indices[..., None, None],
            (
                *batch_shape,
                1,
                self.config['action_dim'],
            ),
        )
        best_actions = jnp.take_along_axis(
            candidate_actions,
            gather_indices,
            axis=candidate_axis,
        )
        best_actions = jnp.squeeze(best_actions, axis=candidate_axis)

        return {
            'raw_actions': raw_candidate_actions,
            'actions': candidate_actions,
            'scores': candidate_scores,
            'best_indices': best_indices,
            'best_actions': best_actions,
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute a TD loss using the best of K next-action candidates."""
        rng, sample_rng = jax.random.split(rng)
        next_candidates = self._generate_candidates(
            batch['next_observations'],
            seed=sample_rng,
        )
        next_actions = jax.lax.stop_gradient(next_candidates['best_actions'])

        next_qs = self.network.select('target_critic')(
            batch['next_observations'],
            actions=next_actions,
        )
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = (
            batch['rewards']
            + self.config['discount'] * batch['masks'] * next_q
        )
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
            'next_candidate_score_mean': next_candidates['scores'].mean(),
            'next_best_score_mean': next_candidates['scores'].max(axis=-1).mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Train all K candidates with BC and the selected one with Q guidance."""
        batch_size, _ = batch['actions'].shape
        _, observation_dim = batch['observations'].shape
        rng, state_noise_rng, time_rng = jax.random.split(rng, 3)

        # Keep the original conditional state-flow matching objective.
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

        # Generate all candidates in one (B, K, ...) batch.  The state flow is
        # evaluated with stored parameters, so Q guidance cannot exploit it.
        rng, candidate_rng = jax.random.split(rng)
        candidates = self._generate_candidates(
            batch['observations'],
            seed=candidate_rng,
            actor_params=grad_params,
        )

        # Every candidate receives behavior-distillation gradients.
        dataset_actions = jnp.expand_dims(batch['actions'], axis=-2)
        distill_loss = jnp.mean(
            jnp.square(candidates['raw_actions'] - dataset_actions)
        )

        # Only the conservatively selected candidate receives actor Q guidance.
        selected_qs = self.network.select('critic')(
            batch['observations'],
            actions=candidates['best_actions'],
        )
        selected_q = selected_qs.mean(axis=0)
        q_loss = -selected_q.mean()
        if self.config['normalize_q_loss']:
            scale = jax.lax.stop_gradient(1 / jnp.abs(selected_q).mean())
            q_loss = scale * q_loss

        actor_loss = (
            state_flow_loss
            + self.config['alpha'] * distill_loss
            + q_loss
        )

        policy_mse = jnp.mean(
            jnp.square(candidates['best_actions'] - batch['actions'])
        )
        return actor_loss, {
            'actor_loss': actor_loss,
            'next_ob_flow_loss': state_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': selected_q.mean(),
            'mse': policy_mse,
            'candidate_score_mean': candidates['scores'].mean(),
            'best_score_mean': candidates['scores'].max(axis=-1).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the multi-candidate PGFQL training objective."""
        info = {}
        rng = rng if rng is not None else self.rng
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

        return critic_loss + actor_loss, info

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
        """Update the multi-candidate actor, critic, and state flow."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Return the best of K batch-generated state-action candidates."""
        del temperature
        candidates = self._generate_candidates(observations, seed=seed)
        return candidates['best_actions']

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
        """Create the original PGFQL networks with candidate batching."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['state_stitch'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

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
            num_candidates=4,
            flow_steps=10,
            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config


# Register through config-module import so main.py needs no changes.
from agents import agents as _agent_registry

_agent_registry['pgfql_candidates'] = PGFQLCandidatesAgent
