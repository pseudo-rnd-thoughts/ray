"""Example showing how to customize an offline data pipeline.

This example:
    - demonstrates how you can customized your offline data pipeline.
    - shows how you can override the `OfflineData` to read raw image
    data and transform it into `numpy ` arrays.
    - explains how you can override the `OfflinePreLearner` to
    transform data further into `SingleAgentEpisode` instances that
    can be processes by the learner connector pipeline.

How to run this script
----------------------
`python [script file name].py --checkpoint-at-end`

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`

Results to expect
-----------------
2024-12-03 19:59:23,043 INFO streaming_executor.py:109 -- Execution plan
of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadBinary] ->
TaskPoolMapOperator[Map(map_to_numpy)] -> LimitOperator[limit=128]
✔️  Dataset execution finished in 10.01 seconds: 100%|███████████████████
███████████████████████████████████████████████████████████████████████|
3.00/3.00 [00:10<00:00, 3.34s/ row]
- ReadBinary->SplitBlocks(11): Tasks: 0; Queued blocks: 0; Resources: 0.0
CPU, 0.0B object store: 100%|█████████████████████████████████████████|
3.00/3.00 [00:10<00:00, 3.34s/ row]
- Map(map_to_numpy): Tasks: 0; Queued blocks: 0; Resources: 0.0 CPU,
0.0B object store: 100%|███████████████████████████████████████████████████|
3.00/3.00 [00:10<00:00, 3.34s/ row]
- limit=128: Tasks: 0; Queued blocks: 0; Resources: 0.0 CPU, 3.0KB object
store: 100%|██████████████████████████████████████████████████████████|
3.00/3.00 [00:10<00:00, 3.34s/ row]
Batch: {'batch': [MultiAgentBatch({}, env_steps=3)]}
"""

import io
import logging
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from PIL import Image
from ray.rllib.algorithms.bc.torch.bc_torch_rl_module import BCTorchRLModule

from ray import data
from ray.actor import ActorHandle
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.bc import BCConfig
from ray.rllib.algorithms.bc.bc_catalog import BCCatalog
from ray.rllib.core.learner.learner import Learner
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import DefaultModelConfig, RLModuleSpec
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.offline.offline_data import OfflineData
from ray.rllib.offline.offline_prelearner import SCHEMA, OfflinePreLearner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import EpisodeType, ModuleID

logger = logging.getLogger(__name__)


class ImageOfflineData(OfflineData):
    """This class overrides `OfflineData` to read in raw image data.

    The image data is from Ray Data`s S3 example bucket, namely
    `ray-example-data/batoidea/JPEGImages/`.
    To read in this data the raw bytes have to be decoded and then
    converted to `numpy` arrays. Each image array has a dimension
    (32, 32, 3).

    To just read in the raw image data and convert it to arrays it
    suffices to override the `OfflineData.__init__` method only.
    Note, that further transformations of the data - specifically
    into `SingleAgentEpisode` data - will be performed in a custom
    `OfflinePreLearner` defined in the `image_offline_prelearner`
    file. You could hard-code the usage of this prelearner here,
    but you will use the `prelearner_class` attribute in the
    `AlgorithmConfig` instead.
    """

    @override(OfflineData)
    def __init__(self, config: AlgorithmConfig):

        # Set class attributes.
        self.config = config
        self.is_multi_agent = self.config.is_multi_agent
        self.materialize_mapped_data = False
        self.path = self.config.input_

        self.data_read_batch_size = self.config.input_read_batch_size
        self.data_is_mapped = False

        # Define your function to map images to numpy arrays.
        def map_to_numpy(row: Dict[str, Any]) -> Dict[str, Any]:
            # Convert to byte stream.
            bytes_stream = io.BytesIO(row["bytes"])
            # Convert to image.
            image = Image.open(bytes_stream)
            # Return an array of the image.
            return {"array": np.array(image)}

        try:
            # Load the dataset and transform to arrays on-the-fly.
            self.data = data.read_binary_files(self.path).map(map_to_numpy)
        except Exception as e:
            logger.error(e)

        # Define further attributes needed in the `sample` method.
        self.batch_iterator = None
        self.map_batches_kwargs = self.config.map_batches_kwargs
        self.iter_batches_kwargs = self.config.iter_batches_kwargs
        # Use a custom OfflinePreLearner if needed.
        self.prelearner_class = self.config.prelearner_class or OfflinePreLearner

        # For remote learner setups.
        self.locality_hints = None
        self.learner_handles = None
        self.module_spec = None


class ImageOfflinePreLearner(OfflinePreLearner):
    """This class transforms image data to `MultiAgentBatch`es.

    While the `ImageOfflineData` class transforms raw image
    bytes to `numpy` arrays, this class maps that data in
    `SingleAgentEpisode` instances through the learner connector
    pipeline and finally outputs a >`MultiAgentBatch` ready for
    training in RLlib's `Learner`s.

    Note, the basic transformation from images to `SingleAgentEpisode`
    instances creates synthetic data that does not rely on any MDP
    and therefore no agent can learn from it. However, this example
    should show how to transform data into this form through
    overriding the `OfflinePreLearner`.
    """

    def __init__(
        self,
        config: "AlgorithmConfig",
        learner: Union[Learner, List[ActorHandle]],
        spaces: Optional[Tuple[gym.Space, gym.Space]] = None,
        module_spec: Optional[MultiRLModuleSpec] = None,
        module_state: Optional[Dict[ModuleID, Any]] = None,
        **kwargs: Dict[str, Any],
    ):
        # Set up necessary class attributes.
        self.config = config
        self.action_space = spaces[1]
        self.observation_space = spaces[0]
        self.input_read_episodes = self.config.input_read_episodes
        self.input_read_sample_batches = self.config.input_read_sample_batches
        self._policies_to_train = "default_policy"
        self._is_multi_agent = False

        # Build the `MultiRLModule` needed for the learner connector.
        self._module = module_spec.build()

        # Build the learner connector pipeline.
        self._learner_connector = self.config.build_learner_connector(
            input_observation_space=self.observation_space,
            input_action_space=self.action_space,
        )

    @override(OfflinePreLearner)
    @staticmethod
    def _map_to_episodes(
        is_multi_agent: bool,
        batch: Dict[str, Union[list, np.ndarray]],
        schema: Dict[str, str] = SCHEMA,
        to_numpy: bool = False,
        input_compress_columns: Optional[List[str]] = None,
        observation_space: gym.Space = None,
        action_space: gym.Space = None,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, List[EpisodeType]]:

        # Define a container for the episodes.
        episodes = []

        # Batches come in as numpy arrays.
        for i, obs in enumerate(batch["array"]):

            # Construct your episode.
            episode = SingleAgentEpisode(
                id_=uuid.uuid4().hex,
                observations=[obs, obs],
                observation_space=observation_space,
                actions=[action_space.sample()],
                action_space=action_space,
                rewards=[random.random()],
                terminated=True,
                truncated=False,
                len_lookback_buffer=0,
                t_started=0,
            )

            # Numpy'ize, if necessary.
            if to_numpy:
                episode.to_numpy()

            # Store the episode in the container.
            episodes.append(episode)

        return {"episodes": episodes}

# Create an Algorithm configuration.
# TODO: Make this an actually running/learning example with RLunplugged
# data from S3 and add this to the CI.
config = (
    BCConfig()
    .environment(
        action_space=gym.spaces.Discrete(2),
        observation_space=gym.spaces.Box(0, 255, (32, 32, 3), np.float32),
    )
    .offline_data(
        input_=["s3://anonymous@ray-example-data/batoidea/JPEGImages/"],
        prelearner_class=ImageOfflinePreLearner,
    )
)

# Specify an `RLModule` and wrap it with a `MultiRLModuleSpec`. Note,
# on `Learner`` side any `RLModule` is an `MultiRLModule`.
module_spec = MultiRLModuleSpec(
    rl_module_specs={
        "default_policy": RLModuleSpec(
            model_config=DefaultModelConfig(
                conv_filters=[[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]],
                conv_activation="relu",
            ),
            inference_only=False,
            module_class=BCTorchRLModule,
            catalog_class=BCCatalog,
            action_space=gym.spaces.Discrete(2),
            observation_space=gym.spaces.Box(0, 255, (32, 32, 3), np.float32),
        ),
    },
)


if __name__ == "__main__":
    # Construct your `OfflineData` class instance.
    offline_data = ImageOfflineData(config)

    # Check, how the data is transformed. Note, the
    # example dataset has only 3 such images.
    batch = offline_data.data.take_batch(3)

    # Construct your `OfflinePreLearner`.
    offline_prelearner = ImageOfflinePreLearner(
        config=config,
        learner=None,
        spaces=(
            config.observation_space,
            config.action_space,
        ),
        module_spec=module_spec,
    )

    # Transform the raw data to `MultiAgentBatch` data.
    batch = offline_prelearner(batch)

    # Show the transformed batch.
    print(f"Batch: {batch}")
