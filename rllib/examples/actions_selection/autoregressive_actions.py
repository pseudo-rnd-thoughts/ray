"""Example on how to define and run with an RLModule with a dependent action space.

This examples:
    - Shows how to write a custom RLModule outputting autoregressive actions.
    The RLModule class used here implements a prior distribution for the first couple
    of actions and then uses the sampled actions to compute the parameters for and
    sample from a posterior distribution.
    - Shows how to configure a PPO algorithm to use the custom RLModule.
    - Stops the training after 100k steps or when the mean episode return
    exceeds -0.012 in evaluation, i.e. if the agent has learned to
    synchronize its actions.

For details on the environment used, take a look at the `CorrelatedActionsEnv`
class. To receive an episode return over 100, the agent must learn how to synchronize
its actions.


How to run this script
----------------------
`python [script file name].py --num-env-runners 2`

Control the number of `EnvRunner`s with the `--num-env-runners` flag. This
will increase the sampling speed.

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
You should reach an episode return of better than -0.5 quickly through a simple PPO
policy. The logic behind beating the env is roughly:

OBS:  optimal a1:   r1:  optimal a2:   r2:
-1      2            0      -1.0        0
-0.5    1/2       -0.5   -0.5/-1.5      0
0       1            0      -1.0        0
0.5     0/1       -0.5   -0.5/-1.5      0
1       0            0      -1.0        0

Meaning, most of the time, you would receive a reward better than -0.5, but worse than
0.0.

+--------------------------------------+------------+--------+------------------+
| Trial name                           | status     |   iter |   total time (s) |
|                                      |            |        |                  |
|--------------------------------------+------------+--------+------------------+
| PPO_CorrelatedActionsEnv_6660d_00000 | TERMINATED |     76 |          132.438 |
+--------------------------------------+------------+--------+------------------+
+------------------------+------------------------+------------------------+
|    episode_return_mean |   num_env_steps_sample |   ...env_steps_sampled |
|                        |             d_lifetime |   _lifetime_throughput |
|------------------------+------------------------+------------------------|
|                  -0.43 |                 152000 |                1283.48 |
+------------------------+------------------------+------------------------+
"""

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import Columns
from ray.rllib.core.distribution.torch.torch_distribution import (
    TorchCategorical,
    TorchDiagGaussian,
    TorchMultiDistribution,
)
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.utils.torch_utils import one_hot
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


class CorrelatedActionsEnv(gym.Env):
    """Environment that can only be solved through an autoregressive action model.

    In each step, the agent observes a random number (between -1 and 1) and has
    to choose two actions, a1 (discrete, 0, 1, or 2) and a2 (cont. between -1 and 1).

    The reward is constructed such that actions need to be correlated to succeed. It's
    impossible for the network to learn each action head separately.

    There are two reward components:
    The first is the negative absolute value of the delta between 1.0 and the sum of
    obs + a1. For example, if obs is -0.3 and a1 was sampled to be 1, then the value of
    the first reward component is:
    r1 = -abs(1.0 - [obs+a1]) = -abs(1.0 - (-0.3 + 1)) = -abs(0.3) = -0.3
    The second reward component is computed as the negative absolute value
    of `obs + a1 + a2`. For example, if obs is 0.5, a1 was sampled to be 0,
    and a2 was sampled to be -0.7, then the value of the second reward component is:
    r2 = -abs(obs + a1 + a2) = -abs(0.5 + 0 - 0.7)) = -abs(-0.2) = -0.2

    Because of this specific reward function, the agent must learn to optimally sample
    a1 based on the observation and to optimally sample a2, based on the observation
    AND the sampled value of a1.

    One way to effectively learn this is through correlated action
    distributions, e.g., in examples/actions/auto_regressive_actions.py

    The game ends after the first step.
    """

    def __init__(self, config=None):
        super().__init__()
        # Observation space (single continuous value between -1. and 1.).
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        # Action space (discrete action a1 and continuous action a2).
        self.action_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(3), gym.spaces.Box(-2.0, 2.0, (1,), np.float32)]
        )

        # Internal state for the environment (e.g., could represent a factor
        # influencing the relationship)
        self.obs = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        """Reset the environment to an initial state."""
        super().reset(seed=seed, options=options)

        # Randomly initialize the observation between -1 and 1.
        self.obs = np.random.uniform(-1, 1, size=(1,))

        return self.obs, {}

    def step(self, action):
        """Apply the autoregressive action and return step information."""

        # Extract individual action components, a1 and a2.
        a1, a2 = action
        a2 = a2[0]  # dissolve shape=(1,)

        # r1 depends on how well a1 is aligned to obs:
        r1 = -abs(1.0 - (self.obs[0] + a1))
        # r2 depends on how well a2 is aligned to both, obs and a1.
        r2 = -abs(self.obs[0] + a1 + a2)

        reward = r1 + r2

        # Optionally: add some noise or complexity to the reward function
        # reward += np.random.normal(0, 0.01)  # Small noise can be added

        # Terminate after each step (no episode length in this simple example)
        return self.obs, reward, True, False, {}


class AutoregressiveActionsRLM(TorchRLModule, ValueFunctionAPI):
    """An RLModule that uses an autoregressive action distribution.

    Actions are sampled in two steps. The first (prior) action component is sampled from
    a categorical distribution. Then, the second (posterior) action component is sampled
    from a posterior distribution that depends on the first action component and the
    other input data (observations).

    Note, this RLModule works in combination with any algorithm, whose Learners require
    the `ValueFunctionAPI`.
    """

    @override(RLModule)
    def setup(self):
        super().setup()

        # Assert the action space is correct.
        assert isinstance(self.action_space, gym.spaces.Tuple)
        assert isinstance(self.action_space[0], gym.spaces.Discrete)
        assert self.action_space[0].n == 3
        assert isinstance(self.action_space[1], gym.spaces.Box)

        self._prior_net = nn.Sequential(
            nn.Linear(
                in_features=self.observation_space.shape[0],
                out_features=256,
            ),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=self.action_space[0].n),
        )

        self._posterior_net = nn.Sequential(
            nn.Linear(
                in_features=self.observation_space.shape[0] + self.action_space[0].n,
                out_features=256,
            ),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=self.action_space[1].shape[0] * 2),
        )

        # Build the value function head.
        self._value_net = nn.Sequential(
            nn.Linear(
                in_features=self.observation_space.shape[0],
                out_features=256,
            ),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=1),
        )

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, TensorType]) -> Dict[str, TensorType]:
        return self._pi(batch[Columns.OBS], inference=True)

    @override(TorchRLModule)
    def _forward_exploration(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        return self._pi(batch[Columns.OBS], inference=False)

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, TensorType]) -> Dict[str, TensorType]:
        return self._forward_exploration(batch)

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, TensorType], embeddings=None):
        # Value function forward pass.
        vf_out = self._value_net(batch[Columns.OBS])
        # Squeeze out last dimension (single node value head).
        return vf_out.squeeze(-1)

    # __sphinx_begin__
    def _pi(self, obs, inference: bool):
        # Prior forward pass and sample a1.
        prior_out = self._prior_net(obs)
        dist_a1 = TorchCategorical.from_logits(prior_out)
        if inference:
            dist_a1 = dist_a1.to_deterministic()
        a1 = dist_a1.sample()

        # Posterior forward pass and sample a2.
        posterior_batch = torch.cat(
            [obs, one_hot(a1, self.action_space[0])],
            dim=-1,
        )
        posterior_out = self._posterior_net(posterior_batch)
        dist_a2 = TorchDiagGaussian.from_logits(posterior_out)
        if inference:
            dist_a2 = dist_a2.to_deterministic()
        a2 = dist_a2.sample()
        actions = (a1, a2)

        # We need logp and distribution parameters for the loss.
        return {
            Columns.ACTION_LOGP: (
                TorchMultiDistribution((dist_a1, dist_a2)).logp(actions)
            ),
            Columns.ACTION_DIST_INPUTS: torch.cat([prior_out, posterior_out], dim=-1),
            Columns.ACTIONS: actions,
        }
        # __sphinx_end__

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        return TorchMultiDistribution.get_partial_dist_cls(
            child_distribution_cls_struct=(TorchCategorical, TorchDiagGaussian),
            input_lens=(3, 2),
        )


parser = add_rllib_example_script_args(
    default_iters=1000,
    default_timesteps=2000000,
    default_reward=-0.45,
)
args = parser.parse_args()

config = (
    PPOConfig()
    .environment(CorrelatedActionsEnv)
    .training(
        train_batch_size_per_learner=2000,
        num_epochs=12,
        minibatch_size=256,
        entropy_coeff=0.005,
        lr=0.0003,
    )
    # Specify the RLModule class to be used.
    .rl_module(
        rl_module_spec=RLModuleSpec(module_class=AutoregressiveActionsRLM),
    )
)


if __name__ == "__main__":
    run_rllib_example_script_experiment(config, args)
