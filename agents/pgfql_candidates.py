import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    ConditionalActorVectorField,
    NextStateVectorField,
    Value,
)


class PGFQLCandidatesAgent(flax.struct.PyTreeNode):
    """PGFQL with transition-density-weighted state candidates.

    A state behavior flow p_beta(s' | s) is pretrained separately and then
    frozen.  During Actor-Critic training it proposes K successor states in a
    single vectorized batch and evaluates their log densities along the same
    flow trajectories.  A temperature-softmax over these log densities gives
    relative support weights.  All K actions receive behavior distillation;
    their Q guidance and the critic target are weighted by state support.
    """

    rng: Any
    network: Any
    state_network: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Stage 1: state behavior flow p_beta(s' | s)
    # ------------------------------------------------------------------
    def state_flow_loss(self, batch, grad_params, rng):
        """Train p_beta(s' | s) with conditional flow matching."""
        batch_size = batch['observations'].shape[0]
        rng, noise_rng, time_rng = jax.random.split(rng, 3)

        state_noises = jax.random.normal(
            noise_rng,
            batch['next_observations'].shape,
        )
        target_next_observations = batch['next_observations']
        times = jax.random.uniform(time_rng, (batch_size, 1))
        interpolated_states = (
            (1 - times) * state_noises
            + times * target_next_observations
        )
        target_velocity = target_next_observations - state_noises
        predicted_velocity = self.state_network.select('state_flow')(
            batch['observations'],
            interpolated_states,
            times,
            params=grad_params,
        )
        loss = jnp.mean(jnp.square(predicted_velocity - target_velocity))
        return loss, {'state_flow/loss': loss}

    @jax.jit
    def state_total_loss(self, batch, grad_params, rng=None):
        """Compute the first-stage state-flow loss."""
        rng = rng if rng is not None else self.rng
        return self.state_flow_loss(batch, grad_params, rng)

    @jax.jit
    def update_state_flow(self, batch):
        """Run one state behavior-flow pretraining update."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.state_total_loss(batch, grad_params, rng=rng)

        new_state_network, info = self.state_network.apply_loss_fn(
            loss_fn=loss_fn
        )
        return self.replace(
            state_network=new_state_network,
            rng=new_rng,
        ), info

    # ------------------------------------------------------------------
    # State-flow log density while sampling
    # ------------------------------------------------------------------
    @staticmethod
    def make_divergence_exact(velocity_apply):
        """Return an exact divergence evaluator for a state vector field."""

        def single_divergence(observation, next_observation, time):
            def velocity_fn(next_observation_input):
                return velocity_apply(
                    observation[None, ...],
                    next_observation_input[None, ...],
                    jnp.reshape(time, (1, 1)),
                )[0]

            jacobian = jax.jacrev(velocity_fn)(next_observation)
            return jnp.trace(jacobian)

        return jax.vmap(single_divergence, in_axes=(0, 0, 0))

    @staticmethod
    def make_divergence_hutchinson(
        velocity_apply,
        probes=1,
        gaussian=False,
    ):
        """Return a Hutchinson trace estimator for a state vector field."""

        def single_divergence(observation, next_observation, time, key):
            def velocity_fn(next_observation_input):
                return velocity_apply(
                    observation[None, ...],
                    next_observation_input[None, ...],
                    jnp.reshape(time, (1, 1)),
                )[0]

            def vector_jacobian_vector(probe_key):
                _, vector_jacobian_product = jax.vjp(
                    velocity_fn,
                    next_observation,
                )
                if gaussian:
                    probe = jax.random.normal(
                        probe_key,
                        next_observation.shape,
                        dtype=next_observation.dtype,
                    )
                else:
                    probe = jax.random.rademacher(
                        probe_key,
                        next_observation.shape,
                        dtype=next_observation.dtype,
                    )
                return jnp.dot(
                    vector_jacobian_product(probe)[0],
                    probe,
                )

            probe_keys = jax.random.split(key, probes)
            return jax.vmap(vector_jacobian_vector)(probe_keys).mean()

        return jax.vmap(single_divergence, in_axes=(0, 0, 0, 0))

    def _sample_next_states_with_logprob(
        self,
        observations,
        seed,
        compute_logprob=True,
    ):
        """Sample K next states and optionally integrate log densities."""
        state_seed, divergence_seed = jax.random.split(seed)
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

        state_dim = self.config['state_dim']
        flat_observations = jnp.reshape(
            candidate_observations,
            (-1, state_dim),
        )
        base_states = jax.random.normal(state_seed, candidate_shape)
        flat_states = jnp.reshape(base_states, (-1, state_dim))

        if self.config['encoder'] is not None:
            encoded_observations = self.state_network.select(
                'state_flow_encoder'
            )(flat_observations)
        else:
            encoded_observations = flat_observations

        def velocity_apply(observation_batch, state_batch, time_batch):
            return self.state_network.select('state_flow')(
                observation_batch,
                state_batch,
                time_batch,
                is_encoded=True,
            )

        if not compute_logprob:
            divergence_fn = None
            divergence_keys = None
        elif self.config['logp_method'] == 'exact':
            divergence_fn = self.make_divergence_exact(velocity_apply)
            divergence_keys = None
        elif self.config['logp_method'] in ('hutch-rade', 'hutch-gaus'):
            divergence_fn = self.make_divergence_hutchinson(
                velocity_apply,
                probes=self.config['logp_hutch_probes'],
                gaussian=(self.config['logp_method'] == 'hutch-gaus'),
            )
            divergence_keys = jax.random.split(
                divergence_seed,
                self.config['flow_steps'] * flat_states.shape[0],
            ).reshape(
                self.config['flow_steps'],
                flat_states.shape[0],
                2,
            )
        else:
            raise ValueError(
                f'Unknown log-density mode: {self.config["logp_method"]}'
            )

        if compute_logprob:
            state_logprob = (
                -0.5 * jnp.sum(jnp.square(flat_states), axis=-1)
                - 0.5 * state_dim * jnp.log(2 * jnp.pi)
            )
        else:
            state_logprob = jnp.zeros(
                (flat_states.shape[0],),
                dtype=flat_states.dtype,
            )
        step_size = 1.0 / self.config['flow_steps']

        def forward_euler_step(step, carry):
            states, logprob = carry
            times = jnp.full(
                (states.shape[0], 1),
                step / self.config['flow_steps'],
                dtype=states.dtype,
            )
            velocities = velocity_apply(
                encoded_observations,
                states,
                times,
            )
            if divergence_fn is None:
                divergences = jnp.zeros_like(logprob)
            elif divergence_keys is None:
                divergences = divergence_fn(
                    encoded_observations,
                    states,
                    times[:, 0],
                )
            else:
                divergences = divergence_fn(
                    encoded_observations,
                    states,
                    times[:, 0],
                    divergence_keys[step],
                )
            return (
                states + velocities * step_size,
                logprob - divergences * step_size,
            )

        flat_next_states, flat_logprobs = jax.lax.fori_loop(
            0,
            self.config['flow_steps'],
            forward_euler_step,
            (flat_states, state_logprob),
        )
        next_states = jnp.reshape(flat_next_states, candidate_shape)
        logprobs = jnp.reshape(
            flat_logprobs,
            (*batch_shape, self.config['num_candidates']),
        )
        return candidate_observations, next_states, logprobs

    def _generate_candidates(
        self,
        observations,
        seed,
        actor_params=None,
        compute_logprob=True,
    ):
        """Generate K state-action candidates and state-support weights."""
        state_seed, action_seed = jax.random.split(seed)
        (
            candidate_observations,
            candidate_next_observations,
            candidate_logprobs,
        ) = self._sample_next_states_with_logprob(
            observations,
            seed=state_seed,
            compute_logprob=compute_logprob,
        )

        observation_ndim = len(self.config['ob_dims'])
        batch_shape = observations.shape[:-observation_ndim]
        action_noises = jax.random.normal(
            action_seed,
            (
                *batch_shape,
                self.config['num_candidates'],
                self.config['action_dim'],
            ),
        )
        raw_candidate_actions = self.network.select(
            'actor_onestep_flow'
        )(
            candidate_observations,
            action_noises,
            candidate_next_observations,
            params=actor_params,
        )
        candidate_actions = jnp.clip(raw_candidate_actions, -1, 1)

        centered_logprobs = candidate_logprobs - jnp.max(
            candidate_logprobs,
            axis=-1,
            keepdims=True,
        )
        candidate_weights = jax.lax.stop_gradient(
            jax.nn.softmax(
                centered_logprobs / self.config['state_temperature'],
                axis=-1,
            )
        )
        return {
            'observations': candidate_observations,
            'next_observations': candidate_next_observations,
            'raw_actions': raw_candidate_actions,
            'actions': candidate_actions,
            'logprobs': candidate_logprobs,
            'weights': candidate_weights,
        }

    def _candidate_weight_metrics(self, candidates):
        weights = candidates['weights']
        entropy = -jnp.sum(
            weights * jnp.log(weights + 1e-8),
            axis=-1,
        )
        return {
            'state_logprob_mean': candidates['logprobs'].mean(),
            'state_logprob_max': candidates['logprobs'].max(axis=-1).mean(),
            'state_weight_max': weights.max(axis=-1).mean(),
            'state_weight_min': weights.min(axis=-1).mean(),
            'state_weight_entropy': entropy.mean(),
        }

    def _best_q_action(self, candidates):
        """Select the highest conservative-ensemble-Q action for evaluation."""
        candidate_qs = self.network.select('critic')(
            candidates['observations'],
            actions=candidates['actions'],
        )
        q_scores = candidate_qs.min(axis=0)
        best_indices = jnp.argmax(q_scores, axis=-1)
        action_dim = self.config['action_dim']
        gather_indices = jnp.broadcast_to(
            best_indices[..., None, None],
            (*best_indices.shape, 1, action_dim),
        )
        best_actions = jnp.take_along_axis(
            candidates['actions'],
            gather_indices,
            axis=-2,
        ).squeeze(axis=-2)
        return best_actions, q_scores

    # ------------------------------------------------------------------
    # Stage 2: Actor-Critic
    # ------------------------------------------------------------------
    def critic_loss(self, batch, grad_params, rng):
        """Compute TD loss with a state-density-weighted candidate target."""
        next_candidates = self._generate_candidates(
            batch['next_observations'],
            seed=rng,
        )
        next_candidate_qs = self.network.select('target_critic')(
            next_candidates['observations'],
            actions=next_candidates['actions'],
        )
        if self.config['q_agg'] == 'min':
            next_candidate_q = next_candidate_qs.min(axis=0)
        else:
            next_candidate_q = next_candidate_qs.mean(axis=0)
        next_q = jnp.sum(
            next_candidates['weights'] * next_candidate_q,
            axis=-1,
        )

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

        metrics = {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'next_candidate_q_mean': next_candidate_q.mean(),
            'next_weighted_q_mean': next_q.mean(),
        }
        metrics.update(self._candidate_weight_metrics(next_candidates))
        return critic_loss, metrics

    def actor_loss(self, batch, grad_params, rng):
        """Distill all K actions and weight their Q guidance by state support."""
        candidates = self._generate_candidates(
            batch['observations'],
            seed=rng,
            actor_params=grad_params,
        )

        dataset_actions = jnp.expand_dims(batch['actions'], axis=-2)
        distill_loss = jnp.mean(
            jnp.square(candidates['raw_actions'] - dataset_actions)
        )

        candidate_qs = self.network.select('critic')(
            candidates['observations'],
            actions=candidates['actions'],
        )
        candidate_q = candidate_qs.mean(axis=0)
        weighted_q = jnp.sum(
            candidates['weights'] * candidate_q,
            axis=-1,
        )
        q_loss = -weighted_q.mean()
        if self.config['normalize_q_loss']:
            scale = jax.lax.stop_gradient(
                1 / (jnp.abs(weighted_q).mean() + 1e-8)
            )
            q_loss = scale * q_loss

        actor_loss = self.config['alpha'] * distill_loss + q_loss
        weighted_action_mse = jnp.sum(
            candidates['weights']
            * jnp.mean(
                jnp.square(
                    candidates['actions'] - dataset_actions
                ),
                axis=-1,
            ),
            axis=-1,
        ).mean()

        metrics = {
            'actor_loss': actor_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'weighted_q': weighted_q.mean(),
            'candidate_q_mean': candidate_q.mean(),
            'mse': weighted_action_mse,
        }
        metrics.update(self._candidate_weight_metrics(candidates))
        return actor_loss, metrics

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the state-weighted multi-candidate PGFQL objective."""
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
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the actor and critic; the state behavior flow stays frozen."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Generate K candidates and return the highest-Q action."""
        del temperature
        candidates = self._generate_candidates(
            observations,
            seed=seed,
            compute_logprob=False,
        )
        best_actions, _ = self._best_q_action(candidates)
        return best_actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create Actor-Critic networks and a separate state behavior flow."""
        if config['state_temperature'] <= 0:
            raise ValueError('state_temperature must be positive')
        if config['num_candidates'] < 1:
            raise ValueError('num_candidates must be at least 1')

        rng = jax.random.PRNGKey(seed)
        rng, init_rng, state_init_rng = jax.random.split(rng, 3)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()
            encoders['state_flow'] = encoder_module()

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
        }
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
        network.params['modules_target_critic'] = (
            network.params['modules_critic']
        )

        state_flow_def = NextStateVectorField(
            hidden_dims=config['state_hidden_dims'],
            state_dim=ob_dims,
            layer_norm=config['state_layer_norm'],
            encoder=encoders.get('state_flow'),
        )
        state_network_info = {
            'state_flow': (
                state_flow_def,
                (ex_observations, ex_observations, ex_times),
            ),
        }
        if encoders.get('state_flow') is not None:
            state_network_info['state_flow_encoder'] = (
                encoders['state_flow'],
                (ex_observations,),
            )
        state_networks = {
            key: value[0] for key, value in state_network_info.items()
        }
        state_network_args = {
            key: value[1] for key, value in state_network_info.items()
        }
        state_network_def = ModuleDict(state_networks)
        state_network_tx = optax.adam(learning_rate=config['lr_state'])
        state_network_params = state_network_def.init(
            state_init_rng,
            **state_network_args,
        )['params']
        state_network = TrainState.create(
            state_network_def,
            state_network_params,
            tx=state_network_tx,
        )

        config['ob_dims'] = ob_dims
        config['state_dim'] = ex_observations.shape[-1]
        config['action_dim'] = action_dim
        return cls(
            rng,
            network=network,
            state_network=state_network,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='pgfql_candidates',
            ob_dims=ml_collections.config_dict.placeholder(list),
            state_dim=ml_collections.config_dict.placeholder(int),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            lr_state=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            state_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            state_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            alpha=1.0,
            num_candidates=10,
            state_flow_epochs=250,
            state_temperature=10.0,
            flow_steps=10,
            logp_method='hutch-rade',
            logp_hutch_probes=1,
            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config


# Register through config-module import so main.py needs no registry changes.
from agents import agents as _agent_registry

_agent_registry['pgfql_candidates'] = PGFQLCandidatesAgent
