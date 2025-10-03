"""Example of how to write a custom APPO that uses a global shared data actor.

The actor is custom code and its remote APIs can be designed as the user requires.
It is created inside the Algorithm's `setup` method and then shared through its
reference with all of the Algorithm's other actors, like EnvRunners, Learners, and
aggregator actors.

During sampling and through using callbacks, each EnvRunner assigns a unique ID
to each sampled episode chunk, then sends manipulated reward data for each sampled
episode chunk to the shared data actor. In particular, the manipulation consists of
each individual reward being multiplied by the EnvRunner's index (from 1 to ...).
Note that the actual reward in the episode is not altered and thus the metrics
reporting continues to show the original reward.

In the learner connector, which creates the train batch from episode data, a custom
connector piece then gets the manipulated rewards from the shared data actor using
the episode chunk's unique ID (see above) and uses the manipulated reward for training.
Note that because of this, different EnvRunners provide different reward signals, which
should make it slightly harder for the value function to learn consistently.
Nevertheless, because the default config here only uses 2 EnvRunners, each multiplying
their rewards by 1 and 2, respectively, this effect is negligible here and the example
should learn how to solve the CartPole-1 env either way.

This example shows:

    - how to write a custom, global shared data actor class with a custom remote API.
    - how an instance of this shared data actor is created upon algorithm
    initialization.
    - how to distribute the actor reference of the shared actor to all other actors
    in the Algorithm, for example EnvRunners, AggregatorActors, and Learners
    - how to subclass an existing algorithm class (APPO) to implement a custom
    Algorithm, overriding the `setup` method to control, which additional actors
    should be created (and shared) by the algo, the `get_state/set_state` methods
    to include the state of the new actor.
    - how - through custom callbacks - the new actor can be written to and queried
    from anywhere within the algorithm, for example its EnvRunner actors or Learners.


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
The experiment should work regardless of whether you are using aggregator
actors or not. By default, the experiment provides one agg. actor per Learner,
but you can set `--num-aggregator-actors-per-learner=0` to have the learner
connector pipeline work directly inside the Learner actor(s).

+-------------------------------------------------+------------+--------+
| Trial name                                      | status     |   iter |
|                                                 |            |        |
|-------------------------------------------------+------------+--------+
| APPOWithSharedDataActor_CartPole-v1_4e860_00000 | TERMINATED |      7 |
+-------------------------------------------------+------------+--------+
+------------------+------------------------+
|   total time (s) |    episode_return_mean |
|                  |                        |
|------------------+------------------------+
|          70.0315 |                 468.42 |
+------------------+------------------------+
"""

import uuid

from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core import Columns
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)

from typing import List

import ray
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.appo import APPO
from ray.rllib.env.env_runner_group import EnvRunnerGroup


@ray.remote
class SharedDataActor:
    """Simple example of an actor that's accessible from all other actors of an algo.

    Exposes remote APIs `put` and `get` to other actors for storing and retrieving
    arbitrary data.
    """

    def __init__(self):
        self.storage = {}

    def get(self, key, delete: bool = False):
        value = self.storage.get(key)
        if delete and key in self.storage:
            del self.storage[key]
        return value

    def put(self, key, value):
        self.storage[key] = value

    def get_state(self):
        return self.storage

    def set_state(self, state):
        self.storage = state


class APPOWithSharedDataActor(APPO):
    def setup(self, config: AlgorithmConfig):
        # Call to parent `setup`.
        super().setup(config)

        # Create shared data actor.
        self.shared_data_actor = SharedDataActor.remote()

        # Share the actor with all other relevant actors.

        def _share(actor, shared_act=self.shared_data_actor):
            actor._shared_data_actor = shared_act
            # Also add shared actor reference to all the learner connector pieces,
            # if applicable.
            if hasattr(actor, "_learner_connector") and actor._learner_connector:
                for conn in actor._learner_connector:
                    conn._shared_data_actor = shared_act

        self.env_runner_group.foreach_env_runner(func=_share)
        if self.eval_env_runner_group:
            self.eval_env_runner_group.foreach_env_runner(func=_share)
        self.learner_group.foreach_learner(func=_share)
        if self._aggregator_actor_manager:
            self._aggregator_actor_manager.foreach_actor(func=_share)

    def get_state(self, *args, **kwargs):
        state = super().get_state(*args, **kwargs)
        # Add shared actor's state.
        state["shared_data_actor"] = ray.get(self.shared_data_actor.get_state.remote())
        return state

    def set_state(self, state, *args, **kwargs):
        super().set_state(state, *args, **kwargs)
        # Set shared actor's state.
        if "shared_data_actor" in state:
            self.shared_data_actor.set_state.remote(state["shared_data_actor"])

    def restore_env_runners(self, env_runner_group: EnvRunnerGroup) -> List[int]:
        restored = super().restore_env_runners(env_runner_group)

        # For the restored EnvRunners, send them the latest shared, global state
        # from the `SharedDataActor`.
        for restored_idx in restored:
            state_ref = self.shared_data_actor.get.remote(
                key=f"EnvRunner_{restored_idx}"
            )
            env_runner_group.foreach_env_runner(
                lambda env_runner, state=state_ref: env_runner._global_state,
                remote_worker_ids=[restored_idx],
                timeout_seconds=0.0,
            )

        return restored


parser = add_rllib_example_script_args(
    default_reward=450.0,
    default_iters=200,
    default_timesteps=2000000,
)
parser.set_defaults(
    num_aggregator_actors_per_learner=1,
)

SPECIAL_REWARDS_KEY = "special_(double)_rewards"
ENV_RUNNER_IDX_KEY = "env_runner_index"
UNIQUE_EPISODE_CHUNK_KEY = "unique_eps_chunk"


# Define 2 simple EnvRunner-based callbacks:


def on_episode_step(*, episode, env_runner, **kwargs):
    # Multiplies the received reward by the env runner index.
    if SPECIAL_REWARDS_KEY not in episode.custom_data:
        episode.custom_data[SPECIAL_REWARDS_KEY] = []
    episode.custom_data[SPECIAL_REWARDS_KEY].append(
        episode.get_rewards(-1) * env_runner.worker_index
    )


def on_sample_end(*, samples, env_runner, **kwargs):
    # Sends the (manipulated) reward sequence to the shared data actor for "pickup" by
    # a Learner. Alternatively, one could also just store the information in the
    # `custom_data` property.
    for episode in samples:
        # Provide a unique key for both episode AND record in the shared
        # data actor.
        unique_key = str(uuid.uuid4())

        # Store the EnvRunner index and unique key in the episode.
        episode.custom_data[ENV_RUNNER_IDX_KEY] = env_runner.worker_index
        episode.custom_data[UNIQUE_EPISODE_CHUNK_KEY] = unique_key

        # Get the manipulated rewards from the episode ..
        special_rewards = episode.custom_data.pop(SPECIAL_REWARDS_KEY)
        # .. and send them under the unique key to the shared data actor.
        env_runner._shared_data_actor.put.remote(
            key=unique_key,
            value=special_rewards,
        )


class ManipulatedRewardConnector(ConnectorV2):
    def __call__(self, *, episodes, batch, metrics, **kwargs):
        if not isinstance(episodes[0], SingleAgentEpisode):
            raise ValueError("This connector only works on `SingleAgentEpisodes`.")
        # Get the manipulated rewards from the shared actor and add them to the train
        # batch.
        for sa_episode in self.single_agent_episode_iterator(episodes):
            unique_key = sa_episode.custom_data[UNIQUE_EPISODE_CHUNK_KEY]
            special_rewards = ray.get(
                self._shared_data_actor.get.remote(unique_key, delete=True)
            )
            if special_rewards is None:
                continue

            assert int(special_rewards[0]) == sa_episode.custom_data[ENV_RUNNER_IDX_KEY]

            # Add one more fake reward, b/c all episodes will be extended
            # (in PPO-style algos) by one artificial timestep for GAE/v-trace
            # computation purposes.
            special_rewards += [0.0]
            self.add_n_batch_items(
                batch=batch,
                column=Columns.REWARDS,
                items_to_add=special_rewards[-len(sa_episode) :],
                num_items=len(sa_episode),
                single_agent_episode=sa_episode,
            )

        return batch


if __name__ == "__main__":
    args = parser.parse_args()

    base_config = (
        APPOConfig(algo_class=APPOWithSharedDataActor)
        .environment("CartPole-v1")
        .callbacks(
            on_episode_step=on_episode_step,
            on_sample_end=on_sample_end,
        )
        .training(
            learner_connector=(lambda obs_sp, act_sp: ManipulatedRewardConnector()),
        )
    )

    run_rllib_example_script_experiment(base_config, args)
