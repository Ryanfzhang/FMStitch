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
    """PGFQL with value-based selection over supported transition candidates.

    The agent factorizes the dataset transition distribution as
    p(a, s' | s) = p(s' | s) p(a | s, s').  The state flow proposes multiple
    next-state candidates, the inverse policy decodes one action per candidate,
    and the critic selects the highest-valued supported action.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss using the selected next action."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self._sample_best_action(
            batch['next_observations'],
            seed=sample_rng,
            num_candidates=self.config['critic_num_candidates'],
        )

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
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
        """Train the state proposal and target-conditioned inverse policy."""
        batch_size, action_dim = batch['actions'].shape
        _, observation_dim = batch['observations'].shape
        rng, state_noise_rng, time_rng = jax.random.split(rng, 3)

        # Learn p(s' | s) with conditional flow matching.
        state_noise = jax.random.normal(state_noise_rng, (batch_size, observation_dim))
        target_next_observations = batch['next_observations']
        times = jax.random.uniform(time_rng, (batch_size, 1))
        interpolated_states = (1 - times) * state_noise + times * target_next_observations
        target_velocity = target_next_observations - state_noise
        predicted_velocity = self.network.select('state_stitch')(
            batch['observations'],
            interpolated_states,
            times,
            params=grad_params,
        )
        state_flow_loss = jnp.mean((predicted_velocity - target_velocity) ** 2)

        # Learn p(a | s, s') from matching dataset tuples.  In particular, do
        # not pair a generated next state with an unrelated dataset action.
        rng, action_noise_rng = jax.random.split(rng)
        action_noise = jax.random.normal(action_noise_rng, (batch_size, action_dim))
        inverse_actions = self.network.select('inverse_policy')(
            batch['observations'],
            action_noise,
            jax.lax.stop_gradient(target_next_observations),
            params=grad_params,
        )
        inverse_loss = jnp.mean((inverse_actions - batch['actions']) ** 2)

        # Keep direct actor guidance optional.  With the default weight of zero,
        # the proposal models remain behavior-supported and policy improvement
        # happens only through candidate selection.
        clipped_actions = jnp.clip(inverse_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=clipped_actions)
        q = jnp.mean(qs, axis=0)
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            scale = jax.lax.stop_gradient(1 / jnp.maximum(jnp.abs(q).mean(), 1e-6))
            q_loss = scale * q_loss

        actor_loss = (
            state_flow_loss
            + self.config['alpha'] * inverse_loss
            + self.config['actor_q_weight'] * q_loss
        )

        return actor_loss, {
            'actor_loss': actor_loss,
            'state_flow_loss': state_flow_loss,
            'inverse_loss': inverse_loss,
            'q_loss': q_loss,
            'q': q.mean(),
        }

    def total_loss(self, batch, grad_params, rng=None):
        """Compute the joint training loss inside the outer update JIT."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for key, value in critic_info.items():
            info[f'critic/{key}'] = value

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for key, value in actor_info.items():
            info[f'actor/{key}'] = value

        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        """Soft-update a target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda parameter, target_parameter: (
                parameter * self.config['tau'] + target_parameter * (1 - self.config['tau'])
            ),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return training metrics."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Generate K transition candidates and return the best action.

        Leading batch dimensions are preserved.  For observations with shape
        (..., observation_dim), the candidate tensors have shape
        (..., num_candidates, feature_dim).
        """
        del temperature
        return self._sample_best_action(
            observations,
            seed=seed,
            num_candidates=self.config['num_candidates'],
        )

    def _sample_best_action(self, observations, seed, num_candidates):
        """Generate and select candidates with a static candidate count."""
        next_state_seed, action_seed = jax.random.split(seed)
        observation_ndim = len(self.config['ob_dims'])
        batch_shape = observations.shape[:-observation_ndim]
        candidate_axis = len(batch_shape)
        candidate_observation_shape = (
            *batch_shape,
            num_candidates,
            *self.config['ob_dims'],
        )

        candidate_observations = jnp.broadcast_to(
            jnp.expand_dims(observations, axis=candidate_axis),
            candidate_observation_shape,
        )

        state_noises = jax.random.normal(next_state_seed, candidate_observation_shape)
        candidate_next_observations = self.compute_flow_next_states(
            candidate_observations,
            noises=state_noises,
        )

        action_noises = jax.random.normal(
            action_seed,
            (*batch_shape, num_candidates, self.config['action_dim']),
        )
        candidate_actions = self.network.select('inverse_policy')(
            candidate_observations,
            action_noises,
            candidate_next_observations,
        )
        candidate_actions = jnp.clip(candidate_actions, -1, 1)

        candidate_qs = self.network.select('critic')(
            candidate_observations,
            actions=candidate_actions,
        )
        if self.config['candidate_q_agg'] == 'min':
            candidate_q = candidate_qs.min(axis=0)
        else:
            candidate_q = candidate_qs.mean(axis=0)

        best_indices = jnp.argmax(candidate_q, axis=-1)
        # ``take_along_axis`` preserves the shape of its indices.  Broadcast
        # the selected candidate index across the full action dimension;
        # otherwise this would return (..., 1) instead of (..., action_dim).
        gather_indices = jnp.broadcast_to(
            best_indices[..., None, None],
            (*best_indices.shape, 1, candidate_actions.shape[-1]),
        )
        best_actions = jnp.take_along_axis(
            candidate_actions,
            gather_indices,
            axis=-2,
        )
        return jnp.squeeze(best_actions, axis=-2)

    def compute_flow_next_states(self, observations, noises):
        """Sample next states inside the caller's outer JIT."""
        if self.config['encoder'] is not None:
            observations = self.network.select('state_stitch_encoder')(observations)

        next_states = noises
        for step in range(self.config['flow_steps']):
            times = jnp.full(
                (*observations.shape[:-1], 1),
                step / self.config['flow_steps'],
            )
            velocities = self.network.select('state_stitch')(
                observations,
                next_states,
                times,
                is_encoded=True,
            )
            next_states = next_states + velocities / self.config['flow_steps']
        return next_states

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a candidate-selection PGFQL agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        if len(ob_dims) != 1:
            raise ValueError(
                'PGFQLCandidatesAgent currently supports vector observations only; '
                f'got observation shape {ob_dims}.'
            )

        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['state_stitch'] = encoder_module()
            encoders['inverse_policy'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        inverse_policy_def = ConditionalActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('inverse_policy'),
        )
        state_stitch_def = NextStateVectorField(
            hidden_dims=config['actor_hidden_dims'],
            state_dim=ob_dims,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('state_stitch'),
        )

        network_info = {
            'critic': (critic_def, (ex_observations, ex_actions)),
            'target_critic': (copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            'inverse_policy': (
                inverse_policy_def,
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
        network = TrainState.create(network_def, network_params, tx=network_tx)

        network.params['modules_target_critic'] = network.params['modules_critic']
        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


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
            actor_q_weight=0.0,
            num_candidates=16,
            critic_num_candidates=1,
            candidate_q_agg='min',
            flow_steps=10,
            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config


# ml_collections loads this file as a config module.  Register the agent as a
# side effect so the existing main.py can use it without modifying the original
# agents/__init__.py.  Run with: --agent=agents/pgfql_candidates.py
from agents import agents as _agent_registry

_agent_registry['pgfql_candidates'] = PGFQLCandidatesAgent
