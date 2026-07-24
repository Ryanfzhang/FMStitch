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
    """Two-stage PGFQL with weighted targets and state-density FAC.

    First pretrain and freeze p_beta(s' | s).  Actor training and rollout use
    the original one-successor PGFQL policy.  The critic target is a
    log-density-softmax weighted sum over K successor/action candidates.  A
    A density-gated hinge penalty lowers Q only when a low-density policy
    candidate is valued above the corresponding dataset action.
    """

    rng: Any
    network: Any
    state_network: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Stage 1: pretrain p_beta(s' | s), then keep it frozen.
    # ------------------------------------------------------------------
    def state_flow_loss(self, batch, grad_params, rng):
        """Train the conditional state flow with flow matching."""
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
        loss = jnp.mean(
            jnp.square(predicted_velocity - target_velocity)
        )
        return loss, {'state_flow/loss': loss}

    @jax.jit
    def state_total_loss(self, batch, grad_params, rng=None):
        """Compute the first-stage state-flow objective."""
        rng = rng if rng is not None else self.rng
        return self.state_flow_loss(batch, grad_params, rng)

    @jax.jit
    def update_state_flow(self, batch):
        """Run one state-flow pretraining update."""
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
    # Frozen state-flow sampling with continuous-flow log density.
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

    def sample_next_states_with_logprob(
        self,
        observations,
        seed,
        compute_logprob=True,
        num_candidates=None,
    ):
        """Sample vectorized successors and integrate their log densities."""
        if num_candidates is None:
            num_candidates = self.config['num_target_candidates']
        state_seed, divergence_seed = jax.random.split(seed)
        batch_shape = observations.shape[:-1]
        candidate_shape = (
            *batch_shape,
            num_candidates,
            self.config['state_dim'],
        )
        candidate_observations = jnp.broadcast_to(
            observations[..., None, :],
            candidate_shape,
        )
        flat_observations = jnp.reshape(
            candidate_observations,
            (-1, self.config['state_dim']),
        )
        base_states = jax.random.normal(state_seed, candidate_shape)
        flat_states = jnp.reshape(
            base_states,
            (-1, self.config['state_dim']),
        )

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
                - 0.5
                * self.config['state_dim']
                * jnp.log(2 * jnp.pi)
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
            (*batch_shape, num_candidates),
        )
        return candidate_observations, next_states, logprobs

    @partial(jax.jit, static_argnames=('mode',))
    def logprob_given_next_states(
        self,
        observations,
        next_observations_final,
        rng=None,
        mode='hutch-rade',
    ):
        """Evaluate log p_beta(s' | s) with the reverse probability ODE."""
        leading_shape = next_observations_final.shape[:-1]
        flat_observations = jnp.reshape(
            observations,
            (-1, self.config['state_dim']),
        )
        flat_next_observations = jnp.reshape(
            next_observations_final,
            (-1, self.config['state_dim']),
        )

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

        if mode == 'exact':
            divergence_fn = self.make_divergence_exact(velocity_apply)
            divergence_keys = None
        elif mode in ('hutch-rade', 'hutch-gaus'):
            if rng is None:
                raise ValueError(
                    'rng is required for Hutchinson divergence'
                )
            divergence_fn = self.make_divergence_hutchinson(
                velocity_apply,
                probes=self.config['logp_hutch_probes'],
                gaussian=(mode == 'hutch-gaus'),
            )
            divergence_keys = jax.random.split(
                rng,
                self.config['flow_steps']
                * flat_next_observations.shape[0],
            ).reshape(
                self.config['flow_steps'],
                flat_next_observations.shape[0],
                2,
            )
        else:
            raise ValueError(f'Unknown log-density mode: {mode}')

        step_size = 1.0 / self.config['flow_steps']
        accumulated_divergence = jnp.zeros(
            (flat_next_observations.shape[0],),
            dtype=flat_next_observations.dtype,
        )

        def reverse_euler_step(step, carry):
            states, log_divergence = carry
            time = (
                self.config['flow_steps'] - step
            ) / self.config['flow_steps']
            times = jnp.full(
                (states.shape[0], 1),
                time,
                dtype=states.dtype,
            )
            velocities = velocity_apply(
                encoded_observations,
                states,
                times,
            )
            if divergence_keys is None:
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
                states - velocities * step_size,
                log_divergence + divergences * step_size,
            )

        base_states, accumulated_divergence = jax.lax.fori_loop(
            0,
            self.config['flow_steps'],
            reverse_euler_step,
            (flat_next_observations, accumulated_divergence),
        )
        base_logprob = (
            -0.5 * jnp.sum(jnp.square(base_states), axis=-1)
            - 0.5
            * self.config['state_dim']
            * jnp.log(2 * jnp.pi)
        )
        logprob = base_logprob - accumulated_divergence
        return jnp.reshape(logprob, leading_shape)

    def generate_candidates(
        self,
        observations,
        seed,
        actor_params=None,
        compute_logprob=True,
        num_candidates=None,
    ):
        """Generate successor/action candidates and logp-softmax weights."""
        if num_candidates is None:
            num_candidates = self.config['num_target_candidates']
        state_seed, action_seed = jax.random.split(seed)
        (
            candidate_observations,
            candidate_next_observations,
            candidate_logprobs,
        ) = self.sample_next_states_with_logprob(
            observations,
            seed=state_seed,
            compute_logprob=compute_logprob,
            num_candidates=num_candidates,
        )
        action_noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[:-1],
                num_candidates,
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

    def critic_loss(self, batch, grad_params, rng):
        """Compute the weighted target plus a density-gated hinge penalty."""
        target_rng, penalty_rng, penalty_logprob_rng = jax.random.split(
            rng, 3
        )

        next_candidates = self.generate_candidates(
            batch['next_observations'],
            seed=target_rng,
        )
        next_qs = self.network.select('target_critic')(
            next_candidates['observations'],
            actions=next_candidates['actions'],
        )
        if self.config['q_agg'] == 'min':
            next_candidate_q = next_qs.min(axis=0)
        else:
            next_candidate_q = next_qs.mean(axis=0)
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
        td_loss = jnp.square(q - target_q).mean()

        penalty_candidates = self.generate_candidates(
            batch['observations'],
            seed=penalty_rng,
            compute_logprob=False,
            num_candidates=self.config['num_penalty_candidates'],
        )
        penalty_actions = jax.lax.stop_gradient(
            penalty_candidates['actions']
        )
        penalty_q = self.network.select('critic')(
            penalty_candidates['observations'],
            actions=penalty_actions,
            params=grad_params,
        )

        if self.config['state_fac_alpha'] > 0:
            penalty_state_logprob = jax.lax.stop_gradient(
                self.logprob_given_next_states(
                    penalty_candidates['observations'],
                    penalty_candidates['next_observations'],
                    rng=penalty_logprob_rng,
                    mode=self.config['logp_method'],
                )
            )
            data_state_logprob = jax.lax.stop_gradient(
                batch['estimated_state_logp']
            )
            logprob_difference = (
                penalty_state_logprob
                - data_state_logprob[:, None]
            )
            state_fac_weights = jax.lax.stop_gradient(
                jnp.clip(
                    jnp.where(
                        logprob_difference < 0,
                        -jnp.expm1(logprob_difference),
                        0.0,
                    ),
                    0.0,
                    self.config['state_fac_weight_max'],
                )
            )
            data_q_reference = jax.lax.stop_gradient(q)[..., None]
            state_fac_q_gap = penalty_q - data_q_reference
            state_fac_hinge = jax.nn.relu(state_fac_q_gap)
            state_fac_penalty = (
                self.config['state_fac_alpha']
                * jnp.mean(
                    state_fac_weights[None, ...]
                    * state_fac_hinge
                )
            )
        else:
            penalty_state_logprob = jnp.zeros(
                penalty_q.shape[1:],
                dtype=penalty_q.dtype,
            )
            data_state_logprob = jnp.zeros_like(batch['rewards'])
            logprob_difference = jnp.zeros_like(
                penalty_state_logprob
            )
            state_fac_weights = jnp.zeros_like(
                penalty_state_logprob
            )
            state_fac_q_gap = jnp.zeros_like(penalty_q)
            state_fac_hinge = jnp.zeros_like(penalty_q)
            state_fac_penalty = jnp.array(0.0, dtype=td_loss.dtype)

        critic_loss = td_loss + state_fac_penalty
        weights = next_candidates['weights']
        weight_entropy = -jnp.sum(
            weights * jnp.log(weights + 1e-8),
            axis=-1,
        )

        return critic_loss, {
            'critic_loss': critic_loss,
            'td_loss': td_loss,
            'next_candidate_q_mean': next_candidate_q.mean(),
            'next_candidate_q_std': next_candidate_q.std(
                axis=-1
            ).mean(),
            'next_weighted_q_mean': next_q.mean(),
            'state_logprob_mean': next_candidates['logprobs'].mean(),
            'state_logprob_max': next_candidates['logprobs'].max(
                axis=-1
            ).mean(),
            'state_weight_max': weights.max(axis=-1).mean(),
            'state_weight_min': weights.min(axis=-1).mean(),
            'state_weight_entropy': weight_entropy.mean(),
            'state_fac_penalty': state_fac_penalty,
            'state_fac_weight_mean': state_fac_weights.mean(),
            'state_fac_weight_max': state_fac_weights.max(),
            'state_fac_active_fraction': (
                state_fac_weights > 0
            ).mean(),
            'state_fac_hinge_active_fraction': (
                (state_fac_weights[None, ...] > 0)
                & (state_fac_q_gap > 0)
            ).mean(),
            'state_fac_q_gap_mean': state_fac_q_gap.mean(),
            'state_fac_hinge_mean': state_fac_hinge.mean(),
            'state_fac_logprob_difference': (
                logprob_difference.mean()
            ),
            'data_state_logprob_mean': data_state_logprob.mean(),
            'penalty_state_logprob_mean': (
                penalty_state_logprob.mean()
            ),
            'penalty_candidate_q': penalty_q.mean(),
            'penalty_candidate_q_max': penalty_q.max(
                axis=-1
            ).mean(),
            'penalty_candidate_action_std': penalty_actions.std(
                axis=1
            ).mean(),
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Keep the original one-successor PGFQL actor update."""
        single_candidate = self.generate_candidates(
            batch['observations'],
            seed=rng,
            actor_params=grad_params,
            compute_logprob=False,
            num_candidates=1,
        )
        raw_actor_actions = single_candidate['raw_actions'].squeeze(
            axis=-2
        )
        actor_actions = single_candidate['actions'].squeeze(axis=-2)
        distill_loss = jnp.mean(
            jnp.square(raw_actor_actions - batch['actions'])
        )

        qs = self.network.select('critic')(
            batch['observations'],
            actions=actor_actions,
        )
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        q_scale = jax.lax.stop_gradient(jnp.abs(q).mean())
        actor_loss = self.config['alpha'] * distill_loss + q_loss
        mse = jnp.mean(
            jnp.square(actor_actions - batch['actions'])
        )

        return actor_loss, {
            'actor_loss': actor_loss,
            'alpha': jnp.asarray(
                self.config['alpha'],
                dtype=distill_loss.dtype,
            ),
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'q_scale': q_scale,
            'mse': mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Keep rollout as one generated successor and one action."""
        del temperature
        single_candidate = self.generate_candidates(
            observations,
            seed=seed,
            compute_logprob=False,
            num_candidates=1,
        )
        return single_candidate['actions'].squeeze(axis=-2)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        if config['state_temperature'] <= 0:
            raise ValueError('state_temperature must be positive')
        if config['num_target_candidates'] < 1:
            raise ValueError('num_target_candidates must be at least 1')
        if config['num_penalty_candidates'] < 1:
            raise ValueError(
                'num_penalty_candidates must be at least 1'
            )
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, state_init_rng = jax.random.split(rng, 3)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]


        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()
            encoders['state_flow'] = encoder_module()

        # Define networks.
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
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions, ex_observations)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

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
        state_network_tx = optax.adam(
            learning_rate=config['lr_state']
        )
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
            agent_name='pgfql_candidates',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            state_dim=ml_collections.config_dict.placeholder(int),  # State dimension (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            lr_state=3e-4,  # State-flow pretraining learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            state_hidden_dims=(512, 512, 512, 512),  # State-flow network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            state_layer_norm=False,  # Whether to use layer normalization for the state flow.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=4.0,  # Shared BC coefficient for all environments.
            state_flow_pretraining=True,  # Pretrain and freeze p_beta(s' | s).
            state_flow_epochs=250,  # Number of state-flow pretraining epochs.
            num_target_candidates=10,  # Candidates in the weighted TD target.
            state_temperature=10.0,  # Temperature for softmax(log p_beta).
            logp_method='hutch-rade',  # Divergence estimator for log density.
            logp_hutch_probes=1,  # Hutchinson probes per state and flow step.
            num_penalty_candidates=4,  # Policy candidates in state-FAC.
            state_fac_alpha=0.1,  # Density-gated Q hinge coefficient.
            state_fac_weight_max=1.0,  # Maximum density-derived weight.
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config


# Register through config-module import so main.py needs no registry changes.
from agents import agents as _agent_registry

_agent_registry['pgfql_candidates'] = PGFQLCandidatesAgent
