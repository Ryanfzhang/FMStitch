import copy
from functools import partial
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
    """PGFQL with behavior-density-selected state-action candidates.

    For each state, the state flow proposes K successor states in parallel and
    the original one-step actor produces K corresponding actions.  A separately
    pretrained and frozen conditional behavior flow filters candidates using
    log p_beta(a | s, s').  The online double critic then selects the highest-Q
    supported candidate.  The same candidate policy is used by actor training,
    critic targets, and environment interaction.
    """

    rng: Any
    network: Any
    bc_network: Any
    logprob_threshold: Any
    config: Any = nonpytree_field()

    def bc_loss(self, batch, grad_params, rng):
        """Train the conditional behavior flow p_beta(a | s, s')."""
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng, time_rng = jax.random.split(rng, 3)

        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_actions = batch['actions']
        times = jax.random.uniform(time_rng, (batch_size, 1))
        interpolated_actions = (
            (1 - times) * noises + times * target_actions
        )
        target_velocity = target_actions - noises
        predicted_velocity = self.bc_network.select('bc_flow')(
            batch['observations'],
            interpolated_actions,
            batch['next_observations'],
            times,
            params=grad_params,
        )
        loss = jnp.mean(jnp.square(predicted_velocity - target_velocity))
        return loss, {'bc/flow_loss': loss}

    @staticmethod
    def make_divergence_exact(velocity_apply):
        """Return an exact divergence evaluator for an action vector field."""

        def single_divergence(observation, next_observation, action, time):
            def velocity_fn(action_input):
                return velocity_apply(
                    observation[None, ...],
                    action_input[None, ...],
                    next_observation[None, ...],
                    jnp.reshape(time, (1, 1)),
                )[0]

            jacobian = jax.jacrev(velocity_fn)(action)
            return jnp.trace(jacobian)

        return jax.vmap(single_divergence, in_axes=(0, 0, 0, 0))

    @staticmethod
    def make_divergence_hutchinson(
        velocity_apply,
        probes=1,
        gaussian=False,
    ):
        """Return a Hutchinson trace estimator for an action vector field."""

        def single_divergence(
            observation,
            next_observation,
            action,
            time,
            key,
        ):
            def velocity_fn(action_input):
                return velocity_apply(
                    observation[None, ...],
                    action_input[None, ...],
                    next_observation[None, ...],
                    jnp.reshape(time, (1, 1)),
                )[0]

            def vector_jacobian_vector(probe_key):
                _, vector_jacobian_product = jax.vjp(velocity_fn, action)
                if gaussian:
                    probe = jax.random.normal(
                        probe_key,
                        action.shape,
                        dtype=action.dtype,
                    )
                else:
                    probe = jax.random.rademacher(
                        probe_key,
                        action.shape,
                        dtype=action.dtype,
                    )
                return jnp.dot(vector_jacobian_product(probe)[0], probe)

            probe_keys = jax.random.split(key, probes)
            return jax.vmap(vector_jacobian_vector)(probe_keys).mean()

        return jax.vmap(single_divergence, in_axes=(0, 0, 0, 0, 0))

    @partial(jax.jit, static_argnames=('mode',))
    def logprob_given_actions(
        self,
        observations,
        next_observations,
        actions_final,
        rng=None,
        mode='exact',
    ):
        """Evaluate log p_beta(a | s, s') with the frozen behavior flow."""
        leading_shape = actions_final.shape[:-1]
        action_dim = actions_final.shape[-1]
        flat_actions = jnp.reshape(actions_final, (-1, action_dim))
        flat_observations = jnp.reshape(
            observations,
            (-1, *self.config['ob_dims']),
        )
        flat_next_observations = jnp.reshape(
            next_observations,
            (-1, *self.config['ob_dims']),
        )

        if self.config['encoder'] is not None:
            encoded_observations = self.bc_network.select('bc_flow_encoder')(
                flat_observations
            )
            encoded_next_observations = self.bc_network.select(
                'bc_flow_encoder'
            )(flat_next_observations)
        else:
            encoded_observations = flat_observations
            encoded_next_observations = flat_next_observations

        def velocity_apply(
            observation_batch,
            action_batch,
            next_observation_batch,
            time_batch,
        ):
            return self.bc_network.select('bc_flow')(
                observation_batch,
                action_batch,
                next_observation_batch,
                time_batch,
                is_encoded=True,
            )

        if mode == 'exact':
            divergence_fn = self.make_divergence_exact(velocity_apply)
            divergence_keys = None
        elif mode in ('hutch-rade', 'hutch-gaus'):
            if rng is None:
                raise ValueError('rng is required for Hutchinson divergence')
            divergence_fn = self.make_divergence_hutchinson(
                velocity_apply,
                probes=self.config['logp_hutch_probes'],
                gaussian=(mode == 'hutch-gaus'),
            )
            divergence_keys = jax.random.split(
                rng,
                self.config['flow_steps'] * flat_actions.shape[0],
            ).reshape(
                self.config['flow_steps'],
                flat_actions.shape[0],
                2,
            )
        else:
            raise ValueError(f'Unknown log-density mode: {mode}')

        integration_steps = self.config['flow_steps']
        step_size = 1.0 / integration_steps
        log_divergence = jnp.zeros(
            (flat_actions.shape[0],),
            dtype=flat_actions.dtype,
        )

        def reverse_euler_step(step, carry):
            actions, accumulated_divergence = carry
            time = (integration_steps - step) / integration_steps
            time_batch = jnp.full(
                (actions.shape[0], 1),
                time,
                dtype=actions.dtype,
            )
            velocities = velocity_apply(
                encoded_observations,
                actions,
                encoded_next_observations,
                time_batch,
            )
            if divergence_keys is None:
                divergences = divergence_fn(
                    encoded_observations,
                    encoded_next_observations,
                    actions,
                    time_batch[:, 0],
                )
            else:
                divergences = divergence_fn(
                    encoded_observations,
                    encoded_next_observations,
                    actions,
                    time_batch[:, 0],
                    divergence_keys[step],
                )
            return (
                actions - velocities * step_size,
                accumulated_divergence + divergences * step_size,
            )

        base_actions, log_divergence = jax.lax.fori_loop(
            0,
            integration_steps,
            reverse_euler_step,
            (flat_actions, log_divergence),
        )
        base_logprob = (
            -0.5 * jnp.sum(jnp.square(base_actions), axis=-1)
            - 0.5 * action_dim * jnp.log(2 * jnp.pi)
        )
        logprob = base_logprob - log_divergence
        return jnp.reshape(logprob, leading_shape)

    def _generate_candidates(self, observations, seed, actor_params=None):
        """Generate K candidates, filter by density, and select by Q."""
        next_state_seed, action_seed, logprob_seed = jax.random.split(seed, 3)
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
        candidate_logprobs = jax.lax.stop_gradient(
            self.logprob_given_actions(
                candidate_observations,
                candidate_next_observations,
                raw_candidate_actions,
                rng=logprob_seed,
                mode=self.config['logp_method'],
            )
        )
        candidate_actions = jnp.clip(raw_candidate_actions, -1, 1)

        candidate_qs = self.network.select('critic')(
            candidate_observations,
            actions=candidate_actions,
        )
        candidate_q_scores = candidate_qs.min(axis=0)
        supported = candidate_logprobs >= self.logprob_threshold
        any_supported = jnp.any(supported, axis=-1)
        supported_q_scores = jnp.where(
            supported,
            candidate_q_scores,
            -jnp.inf,
        )
        best_supported_indices = jnp.argmax(supported_q_scores, axis=-1)
        best_logprob_indices = jnp.argmax(candidate_logprobs, axis=-1)
        best_indices = jnp.where(
            any_supported,
            best_supported_indices,
            best_logprob_indices,
        )
        scalar_gather_indices = best_indices[..., None]
        selected_logprobs = jnp.take_along_axis(
            candidate_logprobs,
            scalar_gather_indices,
            axis=-1,
        ).squeeze(axis=-1)
        selected_q_scores = jnp.take_along_axis(
            candidate_q_scores,
            scalar_gather_indices,
            axis=-1,
        ).squeeze(axis=-1)

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
            'logprobs': candidate_logprobs,
            'q_scores': candidate_q_scores,
            'supported': supported,
            'supported_count': supported.sum(axis=-1),
            'fallback': ~any_supported,
            'all_supported': jnp.all(supported, axis=-1),
            'best_indices': best_indices,
            'best_actions': best_actions,
            'selected_logprobs': selected_logprobs,
            'selected_q_scores': selected_q_scores,
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute TD loss with the highest-density next-action candidate."""
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
            'next_candidate_logprob_mean': next_candidates['logprobs'].mean(),
            'next_best_logprob_mean': next_candidates['logprobs'].max(axis=-1).mean(),
            'next_candidate_q_mean': next_candidates['q_scores'].mean(),
            'next_selected_q_mean': next_candidates[
                'selected_q_scores'
            ].mean(),
            'next_supported_count_mean': next_candidates[
                'supported_count'
            ].mean(),
            'next_fallback_rate': next_candidates['fallback'].astype(
                jnp.float32
            ).mean(),
            'next_all_supported_rate': next_candidates[
                'all_supported'
            ].astype(jnp.float32).mean(),
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

        # Only the behavior-density-selected candidate receives Q guidance.
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
            'candidate_logprob_mean': candidates['logprobs'].mean(),
            'selected_logprob_mean': candidates[
                'selected_logprobs'
            ].mean(),
            'candidate_q_mean': candidates['q_scores'].mean(),
            'selected_q_score_mean': candidates[
                'selected_q_scores'
            ].mean(),
            'supported_count_mean': candidates['supported_count'].mean(),
            'fallback_rate': candidates['fallback'].astype(
                jnp.float32
            ).mean(),
            'all_supported_rate': candidates['all_supported'].astype(
                jnp.float32
            ).mean(),
        }

    @jax.jit
    def bc_total_loss(self, batch, grad_params, rng=None):
        """Compute the first-stage behavior-flow loss."""
        rng = rng if rng is not None else self.rng
        return self.bc_loss(batch, grad_params, rng)

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
    def update_bc(self, batch):
        """Run one behavior-flow pretraining update."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.bc_total_loss(batch, grad_params, rng=rng)

        new_bc_network, info = self.bc_network.apply_loss_fn(
            loss_fn=loss_fn
        )
        return self.replace(bc_network=new_bc_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        """Update the actor, critic, and state flow with frozen BCFlow."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Return the highest-Q candidate above the density threshold."""
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
        """Create PGFQL candidate networks and a separate behavior flow."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, bc_init_rng = jax.random.split(rng, 3)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['state_stitch'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()
            encoders['bc_flow'] = encoder_module()

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

        bc_flow_def = ConditionalActorVectorField(
            hidden_dims=config['bc_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['bc_layer_norm'],
            encoder=encoders.get('bc_flow'),
        )
        bc_network_info = {
            'bc_flow': (
                bc_flow_def,
                (
                    ex_observations,
                    ex_actions,
                    ex_observations,
                    ex_times,
                ),
            ),
        }
        if encoders.get('bc_flow') is not None:
            bc_network_info['bc_flow_encoder'] = (
                encoders['bc_flow'],
                (ex_observations,),
            )
        bc_networks = {
            key: value[0] for key, value in bc_network_info.items()
        }
        bc_network_args = {
            key: value[1] for key, value in bc_network_info.items()
        }
        bc_network_def = ModuleDict(bc_networks)
        bc_network_tx = optax.adam(learning_rate=config['lr_bc'])
        bc_network_params = bc_network_def.init(
            bc_init_rng,
            **bc_network_args,
        )['params']
        bc_network = TrainState.create(
            bc_network_def,
            bc_network_params,
            tx=bc_network_tx,
        )

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(
            rng,
            network=network,
            bc_network=bc_network,
            logprob_threshold=jnp.array(-jnp.inf, dtype=jnp.float32),
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='pgfql_candidates',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            lr_bc=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            bc_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            bc_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            alpha=1.0,
            num_candidates=4,
            bc_epochs=250,
            density_quantile=0.10,
            flow_steps=10,
            logp_method='exact',
            logp_hutch_probes=1,
            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config


# Register through config-module import so main.py needs no changes.
from agents import agents as _agent_registry

_agent_registry['pgfql_candidates'] = PGFQLCandidatesAgent
