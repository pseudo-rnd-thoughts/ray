"""Example of a euclidian-distance curiosity mechanism to learn in sparse-rewards envs.

This example:
    - demonstrates how to define your own euclidian-distance-based curiosity ConnectorV2
    piece that computes intrinsic rewards based on the delta between incoming
    observations and some set of already stored (prior) observations. Thereby, the
    further away the incoming observation is from the already stored ones, the higher
    its corresponding intrinsic reward.
    - shows how this connector piece adds the intrinsic reward to the corresponding
    "main" (extrinsic) reward and overrides the value in the "rewards" key in the
    episode. It thus demonstrates how to do reward shaping in general with RLlib.
    - shows how to plug this connector piece into your algorithm's config.
    - uses Tune and RLlib to learn the env described above and compares 2
    algorithms, one that does use curiosity vs one that does not.

We use the MountainCar-v0 environment, a sparse-reward env that is very hard to learn
for a regular PPO algorithm.


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
from collections import deque
from typing import Any, List, Optional

import gymnasium as gym
import numpy as np

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.env_to_module import MeanStdFilter
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.utils.typing import EpisodeType
from ray.tune.registry import get_trainable_cls


class EuclidianDistanceBasedCuriosity(ConnectorV2):
    """Learner ConnectorV2 piece computing intrinsic rewards with euclidian distance.

    Add this connector piece to your Learner pipeline, through your algo config:
    ```
    config.training(
        learner_connector=lambda obs_sp, act_sp: EuclidianDistanceBasedCuriosity()
    )
    ```

    Intrinsic rewards are computed on the Learner side based on comparing the euclidian
    distance of observations vs already seen ones. A configurable number of observations
    will be stored in a FIFO buffer and all incoming observations have their distance
    measured against those.

    The minimum distance measured is the intrinsic reward for the incoming obs
    (multiplied by a fixed coeffieicnt and added to the "main" extrinsic reward):
    r(i) = intrinsic_reward_coeff * min(ED(o, o(i)) for o in stored_obs))
    where `ED` is the euclidian distance and `stored_obs` is the buffer.

    The intrinsic reward is then added to the extrinsic reward and saved back into the
    episode (under the main "rewards" key).

    Note that the computation and saving back to the episode all happens before the
    actual train batch is generated from the episode data. Thus, the Learner and the
    RLModule used do not take notice of the extra reward added.

    Only one observation per incoming episode will be stored as a new one in the buffer.
    Thereby, we pick the observation with the largest `min(ED)` value over all already
    stored observations to be stored per episode.

    If you would like to use a simpler, count-based mechanism for intrinsic reward
    computations, take a look at the `CountBasedCuriosity` connector piece
    at `ray.rllib.examples.connectors.classes.count_based_curiosity`
    """

    def __init__(
        self,
        input_observation_space: Optional[gym.Space] = None,
        input_action_space: Optional[gym.Space] = None,
        *,
        intrinsic_reward_coeff: float = 1.0,
        max_buffer_size: int = 100,
        **kwargs,
    ):
        """Initializes a CountBasedCuriosity instance.

        Args:
            intrinsic_reward_coeff: The weight with which to multiply the intrinsic
                reward before adding (and saving) it back to the main (extrinsic)
                reward of the episode at each timestep.
        """
        super().__init__(input_observation_space, input_action_space)

        # Create an observation buffer
        self.obs_buffer = deque(maxlen=max_buffer_size)
        self.intrinsic_reward_coeff = intrinsic_reward_coeff

        self._test = 0

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
        if self._test > 10:
            return batch
        self._test += 1
        # Loop through all episodes and change the reward to
        # [reward + intrinsic reward]
        for sa_episode in self.single_agent_episode_iterator(
            episodes=episodes, agents_that_stepped_only=False
        ):
            # Loop through all obs, except the last one.
            observations = sa_episode.get_observations(slice(None, -1))
            # Get all respective (extrinsic) rewards.
            rewards = sa_episode.get_rewards()

            max_dist_obs = None
            max_dist = float("-inf")
            for i, (obs, rew) in enumerate(zip(observations, rewards)):
                # Compare obs to all stored observations and compute euclidian distance.
                min_dist = 0.0
                if self.obs_buffer:
                    min_dist = min(
                        np.sqrt(np.sum((obs - stored_obs) ** 2))
                        for stored_obs in self.obs_buffer
                    )
                if min_dist > max_dist:
                    max_dist = min_dist
                    max_dist_obs = obs

                # Compute our euclidian distance-based intrinsic reward and add it to
                # the main (extrinsic) reward.
                rew += self.intrinsic_reward_coeff * min_dist
                # Store the new reward back to the episode (under the correct
                # timestep/index).
                sa_episode.set_rewards(new_data=rew, at_indices=i)

            # Add the one observation of this episode with the largest (min) euclidian
            # dist to all already stored obs to the buffer (maybe throwing out the
            # oldest obs in there).
            if max_dist_obs is not None:
                self.obs_buffer.append(max_dist_obs)

        return batch


# TODO (sven): SB3's PPO learns MountainCar-v0 until a reward of ~-110.
#  We might have to play around some more with different initializations, etc..
#  to get to these results as well.
parser = add_rllib_example_script_args(
    default_reward=-140.0, default_iters=2000, default_timesteps=1000000
)
parser.set_defaults(
    num_env_runners=4,
)
parser.add_argument(
    "--intrinsic-reward-coeff",
    type=float,
    default=0.0001,
    help="The weight with which to multiply intrinsic rewards before adding them to "
    "the extrinsic ones (default is 0.0001).",
)
parser.add_argument(
    "--no-curiosity",
    action="store_true",
    help="Whether to NOT use count-based curiosity.",
)

args = parser.parse_args()

base_config = (
    get_trainable_cls(args.algo)
    .get_default_config()
    .environment("MountainCar-v0")
    .env_runners(
        env_to_module_connector=lambda env, spaces, device: MeanStdFilter(),
        num_envs_per_env_runner=5,
    )
    .training(
        # The main code in this example: We add the
        # `EuclidianDistanceBasedCuriosity` connector piece to our Learner connector
        # pipeline. This pipeline is fed with collected episodes (either directly
        # from the EnvRunners in on-policy fashion or from a replay buffer) and
        # converts these episodes into the final train batch. The added piece
        # computes intrinsic rewards based on simple observation counts and add them
        # to the "main" (extrinsic) rewards.
        learner_connector=(
            None
            if args.no_curiosity
            else lambda *ags, **kw: EuclidianDistanceBasedCuriosity()
        ),
        # train_batch_size_per_learner=512,
        grad_clip=20.0,
        entropy_coeff=0.003,
        gamma=0.99,
        lr=0.0002,
        lambda_=0.98,
    )
)

if __name__ == "__main__":
    run_rllib_example_script_experiment(base_config, args)
