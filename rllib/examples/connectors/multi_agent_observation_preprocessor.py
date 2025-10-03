"""Example of a ConnectorV2 mapping global observations to n per-module observations.

An RLlib Algorithm has 3 distinct connector pipelines:
- An env-to-module pipeline in an EnvRunner accepting a list of episodes and producing
a batch for an RLModule to compute actions (`forward_inference()` or
`forward_exploration()`).
- A module-to-env pipeline in an EnvRunner taking the RLModule's output and converting
it into an action readable by the environment.
- A learner connector pipeline on a Learner taking a list of episodes and producing
a batch for an RLModule to perform the training forward pass (`forward_train()`).

Each of these pipelines has a fixed set of default ConnectorV2 pieces that RLlib
adds/prepends to these pipelines in order to perform the most basic functionalities.
For example, RLlib adds the `AddObservationsFromEpisodesToBatch` ConnectorV2 into any
env-to-module pipeline to make sure the batch for computing actions contains - at the
minimum - the most recent observation.

On top of these default ConnectorV2 pieces, users can define their own ConnectorV2
pieces (or use the ones available already in RLlib) and add them to one of the 3
different pipelines described above, as required.

This example:
    - shows how the custom `AddOtherAgentsRowIndexToXYPos` ConnectorV2 piece can be
    added to the env-to-module pipeline. It serves as a multi-agent observation
    preprocessor and makes sure than both agents' observations contain necessary
    information about the respective other agent. Without this extra information, the
    agents won't be able to learn to solve the problem optimally.
    - demonstrates that using various such observation mapping connector pieces allows
    users to map from global, multi-agent observations to individual modules'
    observations.


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
You should see the algo reach an episode return of slightly above 20.0, which proves
that both agents learn how to utilize the other agents' row-index (0 or 1) in order
to collide with the other agent and receive an extra +5 reward. Without this collision
during the episode (if one agent reaches its goal, it's removed from the scene and no
collision can occur any longer), the maximum return per agent is under 10.0.

+--------------------------------------+------------+-----------------+--------+
| Trial name                           | status     | loc             |   iter |
|--------------------------------------+------------+-----------------+--------+
| PPO_DoubleRowCorridorEnv_ba678_00000 | TERMINATED | 127.0.0.1:73310 |     37 |
+--------------------------------------+------------+-----------------+--------+
+------------------+-------+-------------------+-------------+-------------+
|   total time (s) |    ts |   combined return |   return p1 |   return p0 |
|------------------+-------+-------------------+-------------+-------------|
|          41.5389 | 19998 |            23.072 |      11.418 |      11.654 |
+------------------+-------+-------------------+-------------+-------------+
"""
from typing import Any

import gymnasium as gym
import numpy as np

from ray.rllib.connectors.env_to_module.flatten_observations import (
    FlattenObservations,
)
from ray.rllib.connectors.env_to_module.observation_preprocessor import (
    MultiAgentObservationPreprocessor,
)
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.utils.typing import AgentID
from ray.tune.registry import get_trainable_cls

torch, _ = try_import_torch()


class DoubleRowCorridorEnv(MultiAgentEnv):
    """A MultiAgentEnv with a single, global observation space for all agents.

    There are two agents in this grid-world-style environment, `agent_0` and `agent_1`.
    The grid has two-rows and multiple columns and agents must, each
    separately, reach their individual goal position to receive a final reward of +10:

    +---------------+
    |0              |
    |              1|
    +---------------+
    Legend:
    0 = agent_0 + goal state for agent_1
    1 = agent_1 + goal state for agent_0

    You can change the length of the grid through providing the "length" key in the
    `config` dict passed to the env's constructor.

    The action space for both agents is Discrete(4), which encodes to moving up, down,
    left, or right in the grid.

    If the two agents collide, meaning they end up in the exact same field after both
    taking their actions at any timestep, an additional reward of +5 is given to both
    agents. Thus, optimal policies aim at seeking the respective other agent first, and
    only then proceeding to their agent's goal position.

    Each agent in the env has an observation space of a 2-tuple containing its own
    x/y-position, where x is the row index, being either 0 (1st row) or 1 (2nd row),
    and y is the column index (starting from 0).
    """

    def __init__(self, config=None):
        super().__init__()

        config = config or {}

        self.length = config.get("length", 15)
        self.terminateds = {}
        self.collided = False

        # Provide information about agents and possible agents.
        self.agents = self.possible_agents = ["agent_0", "agent_1"]
        self.terminateds = {}

        # Observations: x/y, where the first number is the row index, the second number
        # is the column index, for both agents.
        # For example: [0.0, 2.0] means the agent is in row 0 and column 2.
        self._obs_spaces = gym.spaces.Box(
            0.0, self.length - 1, shape=(2,), dtype=np.int32
        )
        self._act_spaces = gym.spaces.Discrete(4)

    @override(MultiAgentEnv)
    def reset(self, *, seed=None, options=None):
        self.agent_pos = {
            "agent_0": [0, 0],
            "agent_1": [1, self.length - 1],
        }
        self.goals = {
            "agent_0": [0, self.length - 1],
            "agent_1": [1, 0],
        }
        self.terminateds = {agent_id: False for agent_id in self.agent_pos}
        self.collided = False

        return self._get_obs(), {}

    @override(MultiAgentEnv)
    def step(self, action_dict):
        rewards = {
            agent_id: -0.1
            for agent_id in self.agent_pos
            if not self.terminateds[agent_id]
        }

        for agent_id, action in action_dict.items():
            row, col = self.agent_pos[agent_id]

            # up
            if action == 0 and row > 0:
                row -= 1
            # down
            elif action == 1 and row < 1:
                row += 1
            # left
            elif action == 2 and col > 0:
                col -= 1
            # right
            elif action == 3 and col < self.length - 1:
                col += 1

            # Update positions.
            self.agent_pos[agent_id] = [row, col]

        obs = self._get_obs()

        # Check for collision (only if both agents are still active).
        if (
            not any(self.terminateds.values())
            and self.agent_pos["agent_0"] == self.agent_pos["agent_1"]
        ):
            if not self.collided:
                rewards["agent_0"] += 5
                rewards["agent_1"] += 5
                self.collided = True

        # Check goals.
        for agent_id in self.agent_pos:
            if (
                self.agent_pos[agent_id] == self.goals[agent_id]
                and not self.terminateds[agent_id]
            ):
                rewards[agent_id] += 10
                self.terminateds[agent_id] = True

        terminateds = {
            agent_id: self.terminateds[agent_id] for agent_id in self.agent_pos
        }
        terminateds["__all__"] = all(self.terminateds.values())

        return obs, rewards, terminateds, {}, {}

    @override(MultiAgentEnv)
    def get_observation_space(self, agent_id: AgentID) -> gym.Space:
        return self._obs_spaces

    @override(MultiAgentEnv)
    def get_action_space(self, agent_id: AgentID) -> gym.Space:
        return self._act_spaces

    def _get_obs(self):
        obs = {}
        pos = self.agent_pos
        for agent_id in self.agent_pos:
            if self.terminateds[agent_id]:
                continue
            obs[agent_id] = np.array(pos[agent_id], dtype=np.int32)
        return obs


class AddOtherAgentsRowIndexToXYPos(MultiAgentObservationPreprocessor):
    """Adds other agent's row index to an x/y-observation for an agent.

    Run this connector with this env:
    :py:class:`~ray.rllib.examples.env.classes.multi_agent.double_row_corridor_env.DoubleRowCorridorEnv`  # noqa

    In this env, 2 agents walk around in a grid-world and must, each separately, reach
    their individual goal position to receive a final reward. However, if they collide
    while search for these goal positions, another larger reward is given to both
    agents. Thus, optimal policies aim at seeking the other agent first, and only then
    proceeding to their agent's goal position.

    Each agents' observation space is a 2-tuple encoding the x/y position
    (x=row, y=column).
    This connector converts these observations to:
    A dict for `agent_0` of structure:
    {
        "agent": Discrete index encoding the position of the agent,
        "other_agent_row": Discrete(2), indicating whether the other agent is in row 0
        or row 1,
    }
    And a 3-tuple for `agent_1`, encoding the x/y position of `agent_1` plus the row
    index (0 or 1) of `agent_0`.

    Note that the row information for the respective other agent, which this connector
    provides, is needed for learning an optimal policy for any of the agents, because
    the env rewards the first collision between the two agents. Hence, an agent needs to
    have information on which row the respective other agent is currently in, so it can
    change to this row and try to collide with this other agent.
    """

    @override(MultiAgentObservationPreprocessor)
    def recompute_output_observation_space(
        self,
        input_observation_space,
        input_action_space,
    ) -> gym.Space:
        """Maps the original (input) observation space to the new one.

        Original observation space is `Dict({agent_n: Box(4,), ...})`.
        Converts the space for `self.agent` into information specific to this agent,
        plus the current row of the respective other agent.
        Output observation space is then:
        `Dict({`agent_n`: Dict(Discrete, Discrete), ...}), where the 1st Discrete
        is the position index of the agent and the 2nd Discrete encodes the current row
        of the other agent (0 or 1). If the other agent is already done with the episode
        (has reached its goal state) a special value of 2 is used.
        """
        agent_0_space = input_observation_space.spaces["agent_0"]
        self._env_corridor_len = agent_0_space.high[1] + 1  # Box.high is inclusive.
        # Env has always 2 rows (and `self._env_corridor_len` columns).
        num_discrete = int(2 * self._env_corridor_len)
        spaces = {
            "agent_0": gym.spaces.Dict(
                {
                    # Exact position of this agent (as an int index).
                    "agent": gym.spaces.Discrete(num_discrete),
                    # Row (0 or 1) of other agent. Or 2, if other agent is already done.
                    "other_agent_row": gym.spaces.Discrete(3),
                }
            ),
            "agent_1": gym.spaces.Box(
                0,
                agent_0_space.high[1],  # 1=column
                shape=(3,),
                dtype=np.float32,
            ),
        }
        return gym.spaces.Dict(spaces)

    @override(MultiAgentObservationPreprocessor)
    def preprocess(self, observations, episode) -> Any:
        # Observations: dict of keys "agent_0" and "agent_1", mapping to the respective
        # x/y positions of these agents (x=row, y=col).
        # For example: [1.0, 4.0] means the agent is in row 1 and column 4.

        new_obs = {}
        # 2=agent is already done
        row_agent_0 = observations.get("agent_0", [2])[0]
        row_agent_1 = observations.get("agent_1", [2])[0]

        if "agent_0" in observations:
            # Compute `agent_0` and `agent_1` enhanced observation.
            index_obs_agent_0 = (
                observations["agent_0"][0] * self._env_corridor_len
                + observations["agent_0"][1]
            )
            new_obs["agent_0"] = {
                "agent": index_obs_agent_0,
                "other_agent_row": row_agent_1,
            }

        if "agent_1" in observations:
            new_obs["agent_1"] = np.array(
                [
                    observations["agent_1"][0],
                    observations["agent_1"][1],
                    row_agent_0,
                ],
                dtype=np.float32,
            )

        return new_obs


parser = add_rllib_example_script_args(
    default_iters=200,
    default_timesteps=200000,
    default_reward=22.0,
)
parser.set_defaults(
    num_agents=2,
)

args = parser.parse_args()

base_config = (
    get_trainable_cls(args.algo)
    .get_default_config()
    .environment(DoubleRowCorridorEnv)
    .env_runners(
        num_envs_per_env_runner=20,
        # Define a list of two connector piece to be prepended to the env-to-module
        # connector pipeline:
        # 1) The custom connector piece: A MultiAgentObservationPreprocessor, which
        # enhances each agents' individual observations through adding the
        # respective other agent's row index to the observation.
        # 2) A FlattenObservations connector to flatten the integer observations
        # for `agent_0`, which the AddOtherAgentsRowIndexToXYPos outputs.
        env_to_module_connector=lambda env, spaces, device: [
            AddOtherAgentsRowIndexToXYPos(),
            # Only flatten agent_0's observations (b/c these are ints that need to
            # be one-hot'd).
            FlattenObservations(multi_agent=True, agent_ids=["agent_0"]),
        ],
    )
    .training(
        train_batch_size_per_learner=512,
        gamma=0.95,
        # Linearly adjust learning rate based on number of GPUs.
        lr=0.0003 * (args.num_learners or 1),
        vf_loss_coeff=0.01,
    )
    .multi_agent(
        policies={"p0", "p1"},
        policy_mapping_fn=lambda aid, eps, **kw: "p0" if aid == "agent_0" else "p1",
    )
)

# PPO specific settings.
if args.algo == "PPO":
    base_config.training(
        minibatch_size=64,
        lambda_=0.1,
        vf_clip_param=10.0,
    )


if __name__ == "__main__":
    run_rllib_example_script_experiment(base_config, args)
