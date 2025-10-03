"""Example of implementing and training with an intrinsic curiosity model (ICM).

This type of curiosity-based learning trains a simplified model of the environment
dynamics based on three networks:
1) Embedding observations into latent space ("feature" network).
2) Predicting the action, given two consecutive embedded observations
("inverse" network).
3) Predicting the next embedded obs, given an obs and action
("forward" network).

The less the ICM is able to predict the actually observed next feature vector,
given obs and action (through the forwards network), the larger the
"intrinsic reward", which will be added to the extrinsic reward of the agent.

Therefore, if a state transition was unexpected, the agent becomes
"curious" and will further explore this transition leading to better
exploration in sparse rewards environments.

For more details, see here:
[1] Curiosity-driven Exploration by Self-supervised Prediction
Pathak, Agrawal, Efros, and Darrell - UC Berkeley - ICML 2017.
https://arxiv.org/pdf/1705.05363.pdf

This example:
    - demonstrates how to write a custom RLModule, representing the ICM from the paper
    above. Note that this custom RLModule does not belong to any individual agent.
    - demonstrates how to write a custom (PPO) TorchLearner that a) adds the ICM to its
    MultiRLModule, b) trains the regular PPO Policy plus the ICM module, using the
    PPO parent loss and the ICM's RLModule's own loss function.

We use a FrozenLake (sparse reward) environment with a custom map size of 12x12 and a
hard time step limit of 22 to make it almost impossible for a non-curiosity based
learners to learn a good policy.


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

Policy using ICM-based curiosity:
+-------------------------------+------------+-----------------+--------+
| Trial name                    | status     | loc             |   iter |
|-------------------------------+------------+-----------------+--------+
| PPO_FrozenLake-v1_52ab2_00000 | TERMINATED | 127.0.0.1:73318 |    392 |
+-------------------------------+------------+-----------------+--------+
+------------------+--------+----------+--------------------+
|   total time (s) |     ts |   reward |   episode_len_mean |
|------------------+--------+----------+--------------------|
|          236.652 | 786000 |      1.0 |               22.0 |
+------------------+--------+----------+--------------------+

Policy NOT using curiosity:
[DOES NOT LEARN AT ALL]
"""
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import tree  # pip install dm_tree

from ray import tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn.torch.dqn_torch_learner import DQNTorchLearner
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.connectors.common.add_observations_from_episodes_to_batch import (
    AddObservationsFromEpisodesToBatch,
)
from ray.rllib.connectors.common.numpy_to_tensor import NumpyToTensor
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.connectors.learner.add_next_observations_from_episodes_to_train_batch import (  # noqa
    AddNextObservationsFromEpisodesToTrainBatch,
)
from ray.rllib.core import DEFAULT_MODULE_ID, Columns
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.rl_module.apis import SelfSupervisedLossAPI
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.utils.torch_utils import one_hot
from ray.rllib.utils.typing import EpisodeType, ModuleID

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.core.learner.torch.torch_learner import TorchLearner

torch, nn = try_import_torch()

ICM_MODULE_ID = "_intrinsic_curiosity_model"


class IntrinsicCuriosityModel(TorchRLModule, SelfSupervisedLossAPI):
    """An intrinsic curiosity model (ICM) as TorchRLModule for better exploration.

    For more details, see:
    [1] Curiosity-driven Exploration by Self-supervised Prediction
    Pathak, Agrawal, Efros, and Darrell - UC Berkeley - ICML 2017.
    https://arxiv.org/pdf/1705.05363.pdf

    Learns a simplified model of the environment based on three networks:
    1) Embedding observations into latent space ("feature" network).
    2) Predicting the action, given two consecutive embedded observations
    ("inverse" network).
    3) Predicting the next embedded obs, given an obs and action
    ("forward" network).

    The less the agent is able to predict the actually observed next feature
    vector, given obs and action (through the forwards network), the larger the
    "intrinsic reward", which will be added to the extrinsic reward.
    Therefore, if a state transition was unexpected, the agent becomes
    "curious" and will further explore this transition leading to better
    exploration in sparse rewards environments.

    .. testcode::

            import numpy as np
            import gymnasium as gym
            import torch

            from ray.rllib.core import Columns
            from ray.rllib.examples.rl_modules.utils.intrinsic_curiosity_model_rlm import (  # noqa
                IntrinsicCuriosityModel
            )

            B = 10  # batch size
            O = 4  # obs (1D) dim
            A = 2  # num actions
            f = 25  # feature dim

            # Construct the RLModule.
            icm_net = IntrinsicCuriosityModel(
                observation_space=gym.spaces.Box(-1.0, 1.0, (O,), np.float32),
                action_space=gym.spaces.Discrete(A),
            )

            # Create some dummy input.
            obs = torch.from_numpy(
                np.random.random_sample(size=(B, O)).astype(np.float32)
            )
            next_obs = torch.from_numpy(
                np.random.random_sample(size=(B, O)).astype(np.float32)
            )
            actions = torch.from_numpy(
                np.random.random_integers(0, A - 1, size=(B,))
            )
            input_dict = {
                Columns.OBS: obs,
                Columns.NEXT_OBS: next_obs,
                Columns.ACTIONS: actions,
            }

            # Call `forward_train()` to get phi (feature vector from obs), next-phi
            # (feature vector from next obs), and the intrinsic rewards (individual, per
            # batch-item forward loss values).
            print(icm_net.forward_train(input_dict))

            # Print out the number of parameters.
            num_all_params = sum(int(np.prod(p.size())) for p in icm_net.parameters())
            print(f"num params = {num_all_params}")
    """

    @override(TorchRLModule)
    def setup(self):
        # Get the ICM achitecture settings from the `model_config` attribute:
        cfg = self.model_config

        feature_dim = cfg.get("feature_dim", 288)

        # Build the feature model (encoder of observations to feature space).
        layers = []
        dense_layers = cfg.get("feature_net_hiddens", (256, 256))
        # `in_size` is the observation space (assume a simple Box(1D)).
        in_size = self.observation_space.shape[0]
        for out_size in dense_layers:
            layers.append(nn.Linear(in_size, out_size))
            if cfg.get("feature_net_activation") not in [None, "linear"]:
                layers.append(
                    get_activation_fn(cfg["feature_net_activation"], "torch")()
                )
            in_size = out_size
        # Last feature layer of n nodes (feature dimension).
        layers.append(nn.Linear(in_size, feature_dim))
        self._feature_net = nn.Sequential(*layers)

        # Build the inverse model (predicting the action between two observations).
        layers = []
        dense_layers = cfg.get("inverse_net_hiddens", (256,))
        # `in_size` is 2x the feature dim.
        in_size = feature_dim * 2
        for out_size in dense_layers:
            layers.append(nn.Linear(in_size, out_size))
            if cfg.get("inverse_net_activation") not in [None, "linear"]:
                layers.append(
                    get_activation_fn(cfg["inverse_net_activation"], "torch")()
                )
            in_size = out_size
        # Last feature layer of n nodes (action space).
        layers.append(nn.Linear(in_size, self.action_space.n))
        self._inverse_net = nn.Sequential(*layers)

        # Build the forward model (predicting the next observation from current one and
        # action).
        layers = []
        dense_layers = cfg.get("forward_net_hiddens", (256,))
        # `in_size` is the feature dim + action space (one-hot).
        in_size = feature_dim + self.action_space.n
        for out_size in dense_layers:
            layers.append(nn.Linear(in_size, out_size))
            if cfg.get("forward_net_activation") not in [None, "linear"]:
                layers.append(
                    get_activation_fn(cfg["forward_net_activation"], "torch")()
                )
            in_size = out_size
        # Last feature layer of n nodes (feature dimension).
        layers.append(nn.Linear(in_size, feature_dim))
        self._forward_net = nn.Sequential(*layers)

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        # Push both observations through feature net to get feature vectors (phis).
        # We cat/batch them here for efficiency reasons (save one forward pass).
        obs = tree.map_structure(
            lambda obs, next_obs: torch.cat([obs, next_obs], dim=0),
            batch[Columns.OBS],
            batch[Columns.NEXT_OBS],
        )
        phis = self._feature_net(obs)
        # Split again to yield 2 individual phi tensors.
        phi, next_phi = torch.chunk(phis, 2)

        # Predict next feature vector (next_phi) with forward model (using obs and
        # actions).
        predicted_next_phi = self._forward_net(
            torch.cat(
                [
                    phi,
                    one_hot(batch[Columns.ACTIONS].long(), self.action_space).float(),
                ],
                dim=-1,
            )
        )

        # Forward loss term: Predicted phi - given phi and action - vs actually observed
        # phi (square-root of L2 norm). Note that this is the intrinsic reward that
        # will be used and the mean of this is the forward net loss.
        forward_l2_norm_sqrt = 0.5 * torch.sum(
            torch.pow(predicted_next_phi - next_phi, 2.0), dim=-1
        )

        output = {
            Columns.INTRINSIC_REWARDS: forward_l2_norm_sqrt,
            # Computed feature vectors (used to compute the losses later).
            "phi": phi,
            "next_phi": next_phi,
        }

        return output

    @override(SelfSupervisedLossAPI)
    def compute_self_supervised_loss(
        self,
        *,
        learner: "TorchLearner",
        module_id: ModuleID,
        config: "AlgorithmConfig",
        batch: Dict[str, Any],
        fwd_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        module = learner.module[module_id].unwrapped()

        # Forward net loss.
        forward_loss = torch.mean(fwd_out[Columns.INTRINSIC_REWARDS])

        # Inverse loss term (predicted action that led from phi to phi' vs
        # actual action taken).
        dist_inputs = module._inverse_net(
            torch.cat([fwd_out["phi"], fwd_out["next_phi"]], dim=-1)
        )
        action_dist = module.get_train_action_dist_cls().from_logits(dist_inputs)

        # Neg log(p); p=probability of observed action given the inverse-NN
        # predicted action distribution.
        inverse_loss = -action_dist.logp(batch[Columns.ACTIONS])
        inverse_loss = torch.mean(inverse_loss)

        # Calculate the ICM loss.
        total_loss = (
            config.learner_config_dict["forward_loss_weight"] * forward_loss
            + (1.0 - config.learner_config_dict["forward_loss_weight"]) * inverse_loss
        )

        learner.metrics.log_dict(
            {
                "mean_intrinsic_rewards": forward_loss,
                "forward_loss": forward_loss,
                "inverse_loss": inverse_loss,
            },
            key=module_id,
            window=1,
        )

        return total_loss

    # Inference and exploration not supported (this is a world-model that should only
    # be used for training).
    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        raise NotImplementedError(
            "`IntrinsicCuriosityModel` should only be used for training! "
            "Only calls to `forward_train()` supported."
        )


class DQNTorchLearnerWithCuriosity(DQNTorchLearner):
    def build(self) -> None:
        super().build()
        add_intrinsic_curiosity_connectors(self)


class PPOTorchLearnerWithCuriosity(PPOTorchLearner):
    def build(self) -> None:
        super().build()
        add_intrinsic_curiosity_connectors(self)


def add_intrinsic_curiosity_connectors(torch_learner: TorchLearner) -> None:
    """Adds two connector pieces to the Learner pipeline, needed for ICM training.

    - The `AddNextObservationsFromEpisodesToTrainBatch` connector makes sure the train
    batch contains the NEXT_OBS for ICM's forward- and inverse dynamics net training.
    - The `IntrinsicCuriosityModelConnector` piece computes intrinsic rewards from the
    ICM and adds the results to the extrinsic reward of the main module's train batch.

    Args:
        torch_learner: The TorchLearner, to whose Learner pipeline the two ICM connector
            pieces should be added.
    """
    learner_config_dict = torch_learner.config.learner_config_dict

    # Assert, we are only training one policy (RLModule) and we have the ICM
    # in our MultiRLModule.
    assert (
        len(torch_learner.module) == 2
        and DEFAULT_MODULE_ID in torch_learner.module
        and ICM_MODULE_ID in torch_learner.module
    )

    # Make sure both curiosity loss settings are explicitly set in the
    # `learner_config_dict`.
    if (
        "forward_loss_weight" not in learner_config_dict
        or "intrinsic_reward_coeff" not in learner_config_dict
    ):
        raise KeyError(
            "When using the IntrinsicCuriosityTorchLearner, both `forward_loss_weight` "
            " and `intrinsic_reward_coeff` must be part of your config's "
            "`learner_config_dict`! Add these values through: `config.training("
            "learner_config_dict={'forward_loss_weight': .., 'intrinsic_reward_coeff': "
            "..})`."
        )

    if torch_learner.config.add_default_connectors_to_learner_pipeline:
        # Prepend a "add-NEXT_OBS-from-episodes-to-train-batch" connector piece
        # (right after the corresponding "add-OBS-..." default piece).
        torch_learner._learner_connector.insert_after(
            AddObservationsFromEpisodesToBatch,
            AddNextObservationsFromEpisodesToTrainBatch(),
        )
        # Append the ICM connector, computing intrinsic rewards and adding these to
        # the main model's extrinsic rewards.
        torch_learner._learner_connector.insert_after(
            NumpyToTensor,
            IntrinsicCuriosityModelConnector(
                intrinsic_reward_coeff=(
                    torch_learner.config.learner_config_dict["intrinsic_reward_coeff"]
                )
            ),
        )


class IntrinsicCuriosityModelConnector(ConnectorV2):
    """Learner ConnectorV2 piece to compute intrinsic rewards based on an ICM.

    For more details, see here:
    [1] Curiosity-driven Exploration by Self-supervised Prediction
    Pathak, Agrawal, Efros, and Darrell - UC Berkeley - ICML 2017.
    https://arxiv.org/pdf/1705.05363.pdf

    This connector piece:
    - requires two RLModules to be present in the MultiRLModule:
    DEFAULT_MODULE_ID (the policy model to be trained) and ICM_MODULE_ID (the instrinsic
    curiosity architecture).
    - must be located toward the end of to your Learner pipeline (after the
    `NumpyToTensor` piece) in order to perform a forward pass on the ICM model with the
    readily compiled batch and a following forward-loss computation to get the intrinsi
    rewards.
    - these intrinsic rewards will then be added to the (extrinsic) rewards in the main
    model's train batch.
    """

    def __init__(
        self,
        input_observation_space: Optional[gym.Space] = None,
        input_action_space: Optional[gym.Space] = None,
        *,
        intrinsic_reward_coeff: float,
        **kwargs,
    ):
        """Initializes a CountBasedCuriosity instance.

        Args:
            intrinsic_reward_coeff: The weight with which to multiply the intrinsic
                reward before adding it to the extrinsic rewards of the main model.
        """
        super().__init__(input_observation_space, input_action_space)

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
        # Assert that the batch is ready.
        assert DEFAULT_MODULE_ID in batch and ICM_MODULE_ID not in batch
        assert (
            Columns.OBS in batch[DEFAULT_MODULE_ID]
            and Columns.NEXT_OBS in batch[DEFAULT_MODULE_ID]
        )
        # TODO (sven): We are performing two forward passes per update right now.
        #  Once here in the connector (w/o grad) to just get the intrinsic rewards
        #  and once in the learner to actually compute the ICM loss and update the ICM.
        #  Maybe we can save one of these, but this would currently harm the DDP-setup
        #  for multi-GPU training.
        with torch.no_grad():
            # Perform ICM forward pass.
            fwd_out = rl_module[ICM_MODULE_ID].forward_train(batch[DEFAULT_MODULE_ID])

        # Add the intrinsic rewards to the main module's extrinsic rewards.
        batch[DEFAULT_MODULE_ID][Columns.REWARDS] += (
            self.intrinsic_reward_coeff * fwd_out[Columns.INTRINSIC_REWARDS]
        )

        # Duplicate the batch such that the ICM also has data to learn on.
        batch[ICM_MODULE_ID] = batch[DEFAULT_MODULE_ID]

        return batch


parser = add_rllib_example_script_args(
    default_iters=2000,
    default_timesteps=10000000,
    default_reward=0.9,
)


class MeasureMaxDistanceToStart(RLlibCallback):
    """Callback measuring the dist of the agent to its start position in FrozenLake-v1.

    Makes the naive assumption that the start position ("S") is in the upper left
    corner of the used map.
    Uses the MetricsLogger to record the (euclidian) distance value.
    """

    def __init__(self):
        super().__init__()
        self.max_dists = defaultdict(float)
        self.max_dists_lifetime = 0.0

    def on_episode_step(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ):
        num_rows = env.envs[0].unwrapped.nrow
        num_cols = env.envs[0].unwrapped.ncol
        obs = np.argmax(episode.get_observations(-1))
        row = obs // num_cols
        col = obs % num_rows
        curr_dist = (row**2 + col**2) ** 0.5
        if curr_dist > self.max_dists[episode.id_]:
            self.max_dists[episode.id_] = curr_dist

    def on_episode_end(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ):
        # Compute current maximum distance across all running episodes
        # (including the just ended one).
        max_dist = max(self.max_dists.values())
        metrics_logger.log_value(
            key="max_dist_travelled_across_running_episodes",
            value=max_dist,
            window=10,
        )
        if max_dist > self.max_dists_lifetime:
            self.max_dists_lifetime = max_dist
        del self.max_dists[episode.id_]

    def on_sample_end(
        self,
        *,
        env_runner,
        metrics_logger,
        samples,
        **kwargs,
    ):
        metrics_logger.log_value(
            key="max_dist_travelled_lifetime",
            value=self.max_dists_lifetime,
            window=1,
        )


args = parser.parse_args()

if args.algo not in ["DQN", "PPO"]:
    raise ValueError("Curiosity example only implemented for either DQN or PPO!")

base_config = (
    tune.registry.get_trainable_cls(args.algo)
    .get_default_config()
    .environment(
        "FrozenLake-v1",
        env_config={
            # Use a 12x12 map.
            "desc": [
                "SFFFFFFFFFFF",
                "FFFFFFFFFFFF",
                "FFFFFFFFFFFF",
                "FFFFFFFFFFFF",
                "FFFFFFFFFFFF",
                "FFFFFFFFFFFF",
                "FFFFFFFFFFFF",
                "FFFFFFFFFFFF",
                "FFFFFFFFFFFF",
                "FFFFFFFFFFFF",
                "FFFFFFFFFFFF",
                "FFFFFFFFFFFG",
            ],
            "is_slippery": False,
            # Limit the number of steps the agent is allowed to make in the env to
            # make it almost impossible to learn without the curriculum.
            "max_episode_steps": 22,
        },
    )
    .callbacks(MeasureMaxDistanceToStart)
    .env_runners(
        num_envs_per_env_runner=5 if args.algo == "PPO" else 1,
        env_to_module_connector=lambda env, spaces, device: FlattenObservations(),
    )
    .training(
        learner_config_dict={
            # Intrinsic reward coefficient.
            "intrinsic_reward_coeff": 0.05,
            # Forward loss weight (vs inverse dynamics loss). Total ICM loss is:
            # L(total ICM) = (
            #     `forward_loss_weight` * L(forward)
            #     + (1.0 - `forward_loss_weight`) * L(inverse_dyn)
            # )
            "forward_loss_weight": 0.2,
        }
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                # The "main" RLModule (policy) to be trained by our algo.
                DEFAULT_MODULE_ID: RLModuleSpec(
                    **(
                        {"model_config": {"vf_share_layers": True}}
                        if args.algo == "PPO"
                        else {}
                    ),
                ),
                # The intrinsic curiosity model.
                ICM_MODULE_ID: RLModuleSpec(
                    module_class=IntrinsicCuriosityModel,
                    # Only create the ICM on the Learner workers, NOT on the
                    # EnvRunners.
                    learner_only=True,
                    # Configure the architecture of the ICM here.
                    model_config={
                        "feature_dim": 288,
                        "feature_net_hiddens": (256, 256),
                        "feature_net_activation": "relu",
                        "inverse_net_hiddens": (256, 256),
                        "inverse_net_activation": "relu",
                        "forward_net_hiddens": (256, 256),
                        "forward_net_activation": "relu",
                    },
                ),
            }
        ),
        # Use a different learning rate for training the ICM.
        algorithm_config_overrides_per_module={
            ICM_MODULE_ID: AlgorithmConfig.overrides(lr=0.0005)
        },
    )
)

# Set PPO-specific hyper-parameters.
if args.algo == "PPO":
    base_config.training(
        num_epochs=6,
        # Plug in the correct Learner class.
        learner_class=PPOTorchLearnerWithCuriosity,
        train_batch_size_per_learner=2000,
        lr=0.0003,
    )
elif args.algo == "DQN":
    base_config.training(
        # Plug in the correct Learner class.
        learner_class=DQNTorchLearnerWithCuriosity,
        train_batch_size_per_learner=128,
        lr=0.00075,
        replay_buffer_config={
            "type": "PrioritizedEpisodeReplayBuffer",
            "capacity": 500000,
            "alpha": 0.6,
            "beta": 0.4,
        },
        # Epsilon exploration schedule for DQN.
        epsilon=[[0, 1.0], [500000, 0.05]],
        n_step=(3, 5),
        double_q=True,
        dueling=True,
    )

success_key = f"{ENV_RUNNER_RESULTS}/max_dist_travelled_across_running_episodes"
stop = {
    success_key: 12.0,
    f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
    NUM_ENV_STEPS_SAMPLED_LIFETIME: args.stop_timesteps,
}


if __name__ == "__main__":
    run_rllib_example_script_experiment(
        base_config,
        args,
        stop=stop,
        success_metric={success_key: stop[success_key]},
    )
