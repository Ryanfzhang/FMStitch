import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, ConditionalActorVectorField, NextStateVectorField


class PGFQLCandidatesAgent(flax.struct.PyTreeNode):
    """PGFQL with a log-density-weighted multi-candidate critic target.

    The actor, state-flow training, rollout policy, and network structure are
    identical to the original PGFQL agent.  Only the critic target changes:
    it samples K state-action candidates in parallel and averages their target
    Q values with a softmax over log p_beta(s' | s).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

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

    def sample_next_state_candidates_with_logprob(
        self,
        observations,
        seed,
    ):
        """Sample K next states and integrate their conditional log density."""
        state_seed, divergence_seed = jax.random.split(seed)
        num_candidates = self.config['num_candidates']
        observation_ndim = len(self.config['ob_dims'])
        batch_shape = observations.shape[:-observation_ndim]
        candidate_axis = len(batch_shape)
        candidate_shape = (
            *batch_shape,
            num_candidates,
            *self.config['ob_dims'],
        )
        candidate_observations = jnp.broadcast_to(
            jnp.expand_dims(observations, axis=candidate_axis),
            candidate_shape,
        )
        base_states = jax.random.normal(state_seed, candidate_shape)

        state_dim = base_states.shape[-1]
        flat_observations = jnp.reshape(
            candidate_observations,
            (-1, state_dim),
        )
        flat_states = jnp.reshape(base_states, (-1, state_dim))

        if self.config['encoder'] is not None:
            encoded_observations = self.network.select(
                'actor_bc_flow_encoder'
            )(flat_observations)
        else:
            encoded_observations = flat_observations

        def velocity_apply(observation_batch, state_batch, time_batch):
            return self.network.select('state_stitch')(
                observation_batch,
                state_batch,
                time_batch,
                is_encoded=True,
            )

        if self.config['logp_method'] == 'exact':
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
                f'Unknown log-density method: '
                f'{self.config["logp_method"]}'
            )

        state_logprob = (
            -0.5 * jnp.sum(jnp.square(flat_states), axis=-1)
            - 0.5 * state_dim * jnp.log(2 * jnp.pi)
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
                states + velocities * step_size,
                logprob - divergences * step_size,
            )

        flat_next_states, flat_logprobs = jax.lax.fori_loop(
            0,
            self.config['flow_steps'],
            forward_euler_step,
            (flat_states, state_logprob),
        )
        candidate_next_states = jnp.reshape(
            flat_next_states,
            candidate_shape,
        )
        candidate_logprobs = jnp.reshape(
            flat_logprobs,
            (*batch_shape, num_candidates),
        )
        return (
            candidate_observations,
            candidate_next_states,
            candidate_logprobs,
        )

    def critic_loss(self, batch, grad_params, rng):
        """Compute the logp-softmax multi-candidate critic loss."""
        state_rng, action_rng = jax.random.split(rng)
        (
            candidate_observations,
            candidate_next_states,
            candidate_logprobs,
        ) = self.sample_next_state_candidates_with_logprob(
            batch['next_observations'],
            seed=state_rng,
        )

        action_noises = jax.random.normal(
            action_rng,
            (
                *batch['next_observations'].shape[:-1],
                self.config['num_candidates'],
                self.config['action_dim'],
            ),
        )
        candidate_actions = self.network.select('actor_onestep_flow')(
            candidate_observations,
            action_noises,
            candidate_next_states,
        )
        candidate_actions = jnp.clip(candidate_actions, -1, 1)

        next_qs = self.network.select('target_critic')(
            candidate_observations,
            actions=candidate_actions,
        )
        if self.config['q_agg'] == 'min':
            candidate_q = next_qs.min(axis=0)
        else:
            candidate_q = next_qs.mean(axis=0)

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
        next_q = jnp.sum(candidate_weights * candidate_q, axis=-1)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        weight_entropy = -jnp.sum(
            candidate_weights * jnp.log(candidate_weights + 1e-8),
            axis=-1,
        )
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'next_candidate_q_mean': candidate_q.mean(),
            'next_weighted_q_mean': next_q.mean(),
            'next_candidate_q_std': candidate_q.std(axis=-1).mean(),
            'state_logprob_mean': candidate_logprobs.mean(),
            'state_logprob_max': candidate_logprobs.max(axis=-1).mean(),
            'state_weight_entropy': weight_entropy.mean(),
            'state_weight_max': candidate_weights.max(axis=-1).mean(),
            'state_weight_min': candidate_weights.min(axis=-1).mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        _, ob_dim = batch['observations'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # next observation loss.
        x_0 = jax.random.normal(x_rng, (batch_size, ob_dim))
        x_1 = batch['next_observations']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select("state_stitch")(batch['observations'], x_t, t, params=grad_params)
        next_ob_flow_loss = jnp.mean((pred - vel) ** 2)

        # action loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, ob_dim))
        target_flow_next_ob = self.compute_flow_next_state(batch['observations'], noises=noises)
        rng, action_noise_rng = jax.random.split(rng)
        noises = jax.random.normal(action_noise_rng, (batch_size, action_dim))
        # actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, batch['next_observations'], params=grad_params)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, target_flow_next_ob, params=grad_params)
        distill_loss = jnp.mean((actor_actions - batch['actions']) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = next_ob_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'next_ob_flow_loss': next_ob_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
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
        """Sample actions from the one-step policy."""
        ob_seed, _ = jax.random.split(seed)

        noises = jax.random.normal(
            ob_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                *self.config['ob_dims'],
            ),
        )
        target_flow_next_ob = self.compute_flow_next_state(observations, noises=noises)
        action_seed, ob_seed = jax.random.split(ob_seed)

        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises, target_flow_next_ob)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_flow_next_state(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        next_state = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('state_stitch')(observations, next_state, t, is_encoded=True)
            next_state = next_state + vels / self.config['flow_steps']
        return next_state

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
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]


        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

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
        # actor_onestep_flow_def = ActorVectorField(
        #     hidden_dims=config['actor_hidden_dims'],
        #     action_dim=action_dim,
        #     layer_norm=config['actor_layer_norm'],
        #     encoder=encoders.get('actor_onestep_flow'),
        # )

        state_stitch_def = NextStateVectorField(
            hidden_dims=config['actor_hidden_dims'],
            state_dim=ob_dims,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions, ex_observations)),
            state_stitch=(state_stitch_def, (ex_observations, ex_observations, ex_times)),
        )
        if encoders.get('actor_onestep_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_onestep_flow_encoder'] = (encoders.get('actor_onestep_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='pgfql_candidates',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=1.0,  # BC coefficient (overridden for selected AntMaze tasks in main.py).
            num_candidates=10,  # Number of critic-target candidates.
            state_temperature=10.0,  # Temperature for logp-softmax target weights.
            flow_steps=10,  # Number of flow steps.
            logp_method='hutch-rade',  # State-flow divergence estimator.
            logp_hutch_probes=1,  # Number of Hutchinson trace probes.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config


# Register through config-module import so main.py needs no registry changes.
from agents import agents as _agent_registry

_agent_registry['pgfql_candidates'] = PGFQLCandidatesAgent
