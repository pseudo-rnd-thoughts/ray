"""Example of handling an Env that outputs protobuf observations.

This example:
    - demonstrates how a custom Env can use protobufs to compress its observation into
    a binary format to save space and gain performance.
    - shows how to use a very simple ConnectorV2 piece that translates these protobuf
    binary observation strings into proper more NN-readable observations (like a 1D
    float32 tensor).

To see more details on which env we are building for this example, take a look at the
`CartPoleWithProtobufObservationSpace` class imported below.
To see more details on which ConnectorV2 piece we are plugging into the config
below, take a look at the `ProtobufCartPoleObservationDecoder` class imported below.


How to run this script
----------------------
`python [script file name].py`

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
You should see results similar to the following in your console output:

+------------------------------------------------------+------------+-----------------+
| Trial name                                           | status     | loc             |
|                                                      |            |                 |
|------------------------------------------------------+------------+-----------------+
| PPO_CartPoleWithProtobufObservationSpace_47dd2_00000 | TERMINATED | 127.0.0.1:67325 |
+------------------------------------------------------+------------+-----------------+
+--------+------------------+------------------------+------------------------+
|   iter |   total time (s) |   episode_return_mean  |   num_episodes_lifetim |
|        |                  |                        |                      e |
+--------+------------------+------------------------+------------------------+
|     17 |          39.9011 |                 513.29 |                    465 |
+--------+------------------+------------------------+------------------------+
"""
from typing import Any, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.examples.envs.utils.cartpole_observations_proto import (
    CartPoleObservation,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.utils.typing import EpisodeType
from ray.tune.registry import get_trainable_cls


class CartPoleWithProtobufObservationSpace(CartPoleEnv):
    """CartPole gym environment that has a protobuf observation space.

    Sometimes, it is more performant for an environment to publish its observations
    as a protobuf message (instead of a heavily nested Dict).

    The protobuf message used here is originally defined in the
    `./utils/cartpole_observations.proto` file. We converted this file into a python
    importable module by compiling it with:

    `protoc --python_out=. cartpole_observations.proto`

    .. which yielded the `cartpole_observations_proto.py` file in the same directory
    (we import this file's `CartPoleObservation` message here).

    The new observation space is a (binary) Box(0, 255, ([len of protobuf],), uint8).

    A ConnectorV2 pipeline or simpler gym.Wrapper will have to be used to convert this
    observation format into an NN-readable (e.g. float32) 1D tensor.
    """

    def __init__(self, config=None):
        super().__init__()
        dummy_obs = self._convert_observation_to_protobuf(
            np.array([1.0, 1.0, 1.0, 1.0])
        )
        bin_length = len(dummy_obs)
        self.observation_space = gym.spaces.Box(0, 255, (bin_length,), np.uint8)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        proto_observation = self._convert_observation_to_protobuf(observation)
        return proto_observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        proto_observation = self._convert_observation_to_protobuf(observation)
        return proto_observation, info

    def _convert_observation_to_protobuf(self, observation):
        x_pos, x_veloc, angle_pos, angle_veloc = observation

        # Create the Protobuf message
        cartpole_observation = CartPoleObservation()
        cartpole_observation.x_pos = x_pos
        cartpole_observation.x_veloc = x_veloc
        cartpole_observation.angle_pos = angle_pos
        cartpole_observation.angle_veloc = angle_veloc

        # Serialize to binary string.
        return np.frombuffer(cartpole_observation.SerializeToString(), np.uint8)


# if __name__ == "__main__":
#     env = CartPoleWithProtobufObservationSpace()
#     obs, info = env.reset()
#
#     # Test loading a protobuf object with data from the obs binary string
#     # (uint8 ndarray).
#     byte_str = obs.tobytes()
#     obs_protobuf = CartPoleObservation()
#     obs_protobuf.ParseFromString(byte_str)
#     print(obs_protobuf)
#
#     terminated = truncated = False
#     while not terminated and not truncated:
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)
#
#         print(obs)


class ProtobufCartPoleObservationDecoder(ConnectorV2):
    """Env-to-module ConnectorV2 piece decoding protobuf obs into CartPole-v1 obs.

    Add this connector piece to your env-to-module pipeline, through your algo config:
    ```
    config.env_runners(
        env_to_module_connector=(
            lambda env, spaces, device: ProtobufCartPoleObservationDecoder()
        )
    )
    ```

    The incoming observation space must be a 1D Box of dtype uint8
    (which is the same as a binary string). The outgoing observation space is the
    normal CartPole-v1 1D space: Box(-inf, inf, (4,), float32).
    """

    @override(ConnectorV2)
    def recompute_output_observation_space(
        self,
        input_observation_space: gym.Space,
        input_action_space: gym.Space,
    ) -> gym.Space:
        # Make sure the incoming observation space is a protobuf (binary string).
        assert (
            isinstance(input_observation_space, gym.spaces.Box)
            and len(input_observation_space.shape) == 1
            and input_observation_space.dtype.name == "uint8"
        )
        # Return CartPole-v1's natural observation space.
        return gym.spaces.Box(float("-inf"), float("inf"), (4,), np.float32)

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
        # Loop through all episodes and change the observation from a binary string
        # to an actual 1D np.ndarray (normal CartPole-v1 obs).
        for sa_episode in self.single_agent_episode_iterator(episodes=episodes):
            # Get last obs (binary string).
            obs = sa_episode.get_observations(-1)
            obs_bytes = obs.tobytes()
            obs_protobuf = CartPoleObservation()
            obs_protobuf.ParseFromString(obs_bytes)

            # Set up the natural CartPole-v1 observation tensor from the protobuf
            # values.
            new_obs = np.array(
                [
                    obs_protobuf.x_pos,
                    obs_protobuf.x_veloc,
                    obs_protobuf.angle_pos,
                    obs_protobuf.angle_veloc,
                ],
                np.float32,
            )

            # Write the new observation (1D tensor) back into the Episode.
            sa_episode.set_observations(new_data=new_obs, at_indices=-1)

        # Return `data` as-is.
        return batch


parser = add_rllib_example_script_args(default_timesteps=200000, default_reward=400.0)


args = parser.parse_args()

base_config = (
    get_trainable_cls(args.algo).get_default_config()
    # Set up the env to be CartPole-v1, but with protobuf observations.
    .environment(CartPoleWithProtobufObservationSpace)
    # Plugin our custom ConnectorV2 piece to translate protobuf observations
    # (box of dtype uint8) into NN-readible ones (1D tensor of dtype flaot32).
    .env_runners(
        env_to_module_connector=(
            lambda env, spaces, device: ProtobufCartPoleObservationDecoder()
        ),
    )
)


if __name__ == "__main__":
    run_rllib_example_script_experiment(base_config, args)
