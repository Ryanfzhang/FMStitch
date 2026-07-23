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
    """PGFQL with a random-successor conservative margin.

    The actor and Bellman target are identical to the original PGFQL.  The
    only additional critic term penalizes actions induced by randomly
    mismatched successor states when their Q values exceed the dataset-action
    Q values by more than the configured margin.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def sample_random_successor_actions(self, batch, rng):
        """Generate actions conditioned on randomly mismatched successors."""
        random_state_rng, random_action_rng = jax.random.split(rng)
        batch_size, observation_dim = batch['observations'].shape
        num_samples = self.config['num_random_samples']
        action_dim = self.config['action_dim']

        repeated_observations = jnp.broadcast_to(
            batch['observations'][:, None, :],
            (batch_size, num_samples, observation_dim),
        )

        # Random stitching uses valid successor states sampled independently
        # from the batch marginal instead of unphysical uniform state vectors.
        random_indices = jax.random.randint(
            random_state_rng,
            (batch_size, num_samples),
            minval=0,
            maxval=batch_size,
        )
        random_next_observations = batch['next_observations'][
            random_indices
        ]
        random_action_noises = jax.random.normal(
            random_action_rng,
            (batch_size, num_samples, action_dim),
        )
        random_conditioned_actions = self.network.select(
            'actor_onestep_flow'
        )(
            repeated_observations,
            random_action_noises,
            random_next_observations,
        )
        return jax.lax.stop_gradient(
            jnp.clip(random_conditioned_actions, -1, 1)
        )

    def critic_loss(self, batch, grad_params, rng):
        """Compute original TD plus a random-successor hinge penalty."""
        target_rng, penalty_rng = jax.random.split(rng)

        # Keep the original PGFQL one-sample Bellman target.
        next_actions = self.sample_actions(
            batch['next_observations'],
            seed=target_rng,
        )
        next_actions = jnp.clip(next_actions, -1, 1)

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
        td_loss = jnp.square(q - target_q).mean()

        random_actions = self.sample_random_successor_actions(
            batch,
            penalty_rng,
        )
        random_observations = jnp.broadcast_to(
            batch['observations'][:, None, :],
            (
                batch['observations'].shape[0],
                self.config['num_random_samples'],
                batch['observations'].shape[-1],
            ),
        )
        random_q = self.network.select('critic')(
            random_observations,
            actions=random_actions,
            params=grad_params,
        )
        margin_violation = (
            random_q
            - q[..., None]
            + self.config['random_penalty_margin']
        )
        hinge_loss = jax.nn.relu(margin_violation).mean()
        random_penalty = (
            self.config['random_penalty_alpha'] * hinge_loss
        )
        critic_loss = td_loss + random_penalty

        return critic_loss, {
            'critic_loss': critic_loss,
            'td_loss': td_loss,
            'random_penalty': random_penalty,
            'random_hinge_loss': hinge_loss,
            'random_margin_violation': margin_violation.mean(),
            'random_penalty_active_fraction': (
                margin_violation > 0
            ).mean(),
            'random_successor_q': random_q.mean(),
            'random_successor_q_max': random_q.max(axis=-1).mean(),
            'random_successor_action_std': random_actions.std(
                axis=1
            ).mean(),
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
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

        # Keep the original PGFQL actor: both distillation and Q improvement
        # use the same successor sampled from the current state flow.
        rng, state_rng, policy_action_rng = jax.random.split(rng, 3)
        state_noises = jax.random.normal(
            state_rng,
            (batch_size, ob_dim),
        )
        policy_next_observations = self.compute_flow_next_state(
            batch['observations'],
            noises=state_noises,
        )
        policy_action_noises = jax.random.normal(
            policy_action_rng,
            (batch_size, action_dim),
        )
        actor_actions = self.network.select('actor_onestep_flow')(
            batch['observations'],
            policy_action_noises,
            policy_next_observations,
            params=grad_params,
        )
        distill_loss = jnp.mean(
            jnp.square(actor_actions - batch['actions'])
        )

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(
            batch['observations'],
            actions=actor_actions,
        )
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
            num_random_samples=4,  # Randomly mismatched successors per state.
            random_penalty_alpha=0.1,  # Random-successor hinge weight.
            random_penalty_margin=1.0,  # Desired Q margin below dataset actions.
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config


# Register through config-module import so main.py needs no registry changes.
from agents import agents as _agent_registry

_agent_registry['pgfql_candidates'] = PGFQLCandidatesAgent
