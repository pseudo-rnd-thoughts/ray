"""Example of using a count-based curiosity mechanism to learn in sparse-rewards envs.

This example:
    - demonstrates how to define your own count-based curiosity ConnectorV2 piece
    that computes intrinsic rewards based on simple observation counts and adds these
    intrinsic rewards to the "main" (extrinsic) rewards.
    - shows how this connector piece overrides the main (extrinsic) rewards in the
    episode and thus demonstrates how to do reward shaping in general with RLlib.
    - shows how to plug this connector piece into your algorithm's config.
    - uses Tune and RLlib to learn the env described above and compares 2
    algorithms, one that does use curiosity vs one that does not.

We use a FrozenLake (sparse reward) environment with a map size of 8x8 and a time step
limit of 14 to make it almost impossible for a non-curiosity based policy to learn.


How to run this script
----------------------
`python [script file name].py`

Use the `--no-curiosity` flag to disable curiosity learning and force your policy
to be trained on the task w/o the use of intrinsic rewards. With this option, the
algorithm should NOT succeed.

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
In the console output, you can see that only a PPO policy that uses curiosity can
actually learn.

Policy using count-based curiosity:
+-------------------------------+------------+--------+------------------+
| Trial name                    | status     |   iter |   total time (s) |
|                               |            |        |                  |
|-------------------------------+------------+--------+------------------+
| PPO_FrozenLake-v1_109de_00000 | TERMINATED |     48 |            44.46 |
+-------------------------------+------------+--------+------------------+
+------------------------+-------------------------+------------------------+
|    episode_return_mean |   num_episodes_lifetime |   num_env_steps_traine |
|                        |                         |             d_lifetime |
|------------------------+-------------------------+------------------------|
|                   0.99 |                   12960 |                 194000 |
+------------------------+-------------------------+------------------------+

Policy NOT using curiosity:
[DOES NOT LEARN AT ALL]
"""
from collections import Counter
from typing import Any, List, Optional

import gymnasium as gym

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.utils.typing import EpisodeType
from ray.tune.registry import get_trainable_cls


class CountBasedCuriosity(ConnectorV2):
    """Learner ConnectorV2 piece to compute intrinsic rewards based on obs counts.

    Add this connector piece to your Learner pipeline, through your algo config:
    ```
    config.training(
        learner_connector=lambda obs_sp, act_sp: CountBasedCuriosity()
    )
    ```

    Intrinsic rewards are computed on the Learner side based on naive observation
    counts, which is why this connector should only be used for simple environments
    with a reasonable number of possible observations. The intrinsic reward for a given
    timestep is:
    r(i) = intrinsic_reward_coeff * (1 / C(obs(i)))
    where C is the total (lifetime) count of the obs at timestep i.

    The intrinsic reward is added to the extrinsic reward and saved back into the
    episode (under the main "rewards" key).

    Note that the computation and saving back to the episode all happens before the
    actual train batch is generated from the episode data. Thus, the Learner and the
    RLModule used do not take notice of the extra reward added.

    If you would like to use a more sophisticated mechanism for intrinsic reward
    computations, take a look at the `EuclidianDistanceBasedCuriosity` connector piece
    at `ray.rllib.examples.connectors.classes.euclidian_distance_based_curiosity`
    """

    def __init__(
        self,
        input_observation_space: Optional[gym.Space] = None,
        input_action_space: Optional[gym.Space] = None,
        *,
        intrinsic_reward_coeff: float = 1.0,
        **kwargs,
    ):
        """Initializes a CountBasedCuriosity instance.

        Args:
            intrinsic_reward_coeff: The weight with which to multiply the intrinsic
                reward before adding (and saving) it back to the main (extrinsic)
                reward of the episode at each timestep.
        """
        super().__init__(input_observation_space, input_action_space)

        # Naive observation counter.
        self._counts = Counter()
        self.intrinsic_reward_coeff = intrinsic_reward_coeff

    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Any,
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        # Loop through all episodes and change the reward to
        # [reward + intrinsic reward]
        for sa_episode in self.single_agent_episode_iterator(
            episodes=episodes, agents_that_stepped_only=False
        ):
            # Loop through all observations, except the last one.
            observations = sa_episode.get_observations(slice(None, -1))
            # Get all respective extrinsic rewards.
            rewards = sa_episode.get_rewards()

            for i, (obs, rew) in enumerate(zip(observations, rewards)):
                # Add 1 to obs counter.
                obs = tuple(obs)
                self._counts[obs] += 1
                # Compute the count-based intrinsic reward and add it to the extrinsic
                # reward.
                rew += self.intrinsic_reward_coeff * (1 / self._counts[obs])
                # Store the new reward back to the episode (under the correct
                # timestep/index).
                sa_episode.set_rewards(new_data=rew, at_indices=i)

        return batch


parser = add_rllib_example_script_args(
    default_reward=0.99, default_iters=200, default_timesteps=1000000
)
parser.add_argument(
    "--intrinsic-reward-coeff",
    type=float,
    default=1.0,
    help="The weight with which to multiply intrinsic rewards before adding them to "
    "the extrinsic ones (default is 1.0).",
)
parser.add_argument(
    "--no-curiosity",
    action="store_true",
    help="Whether to NOT use count-based curiosity.",
)

ENV_OPTIONS = {
    "is_slippery": False,
    # Use this hard-to-solve 8x8 map with lots of holes (H) to fall into and only very
    # few valid paths from the starting state (S) to the goal state (G).
    "desc": [
        "SFFHFFFH",
        "FFFHFFFF",
        "FFFHHFFF",
        "FFFFFFFH",
        "HFFHFFFF",
        "HHFHFFHF",
        "FFFHFHHF",
        "FHFFFFFG",
    ],
    # Limit the number of steps the agent is allowed to make in the env to
    # make it almost impossible to learn without (count-based) curiosity.
    "max_episode_steps": 14,
}


args = parser.parse_args()

base_config = (
    get_trainable_cls(args.algo)
    .get_default_config()
    .environment(
        "FrozenLake-v1",
        env_config=ENV_OPTIONS,
    )
    .env_runners(
        num_envs_per_env_runner=5,
        # Flatten discrete observations (into one-hot vectors).
        env_to_module_connector=lambda env, spaces, device: FlattenObservations(),
    )
    .training(
        # The main code in this example: We add the `CountBasedCuriosity` connector
        # piece to our Learner connector pipeline.
        # This pipeline is fed with collected episodes (either directly from the
        # EnvRunners in on-policy fashion or from a replay buffer) and converts
        # these episodes into the final train batch. The added piece computes
        # intrinsic rewards based on simple observation counts and add them to
        # the "main" (extrinsic) rewards.
        learner_connector=(
            None if args.no_curiosity else lambda *ags, **kw: CountBasedCuriosity()
        ),
        num_epochs=10,
        vf_loss_coeff=0.01,
    )
    .rl_module(model_config=DefaultModelConfig(vf_share_layers=True))
)


if __name__ == "__main__":
    run_rllib_example_script_experiment(base_config, args)
