from typing_extensions import Self

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.algorithms.custom.utils.vpg import VPG
from ray.rllib.utils.annotations import override


class VPGConfig(AlgorithmConfig):
    """A simple VPG (vanilla policy gradient) algorithm w/o value function support.

    Use for testing purposes only!

    This Algorithm should use the VPGTorchLearner and VPGTorchRLModule
    """

    # A test setting to activate metrics on mean weights.
    report_mean_weights: bool = True

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or VPG)

        # VPG specific settings.
        self.num_episodes_per_train_batch = 10
        # Note that we don't have to set this here, because we tell the EnvRunners
        # explicitly to sample entire episodes. However, for good measure, we change
        # this setting here either way.
        self.batch_mode = "complete_episodes"

        # VPG specific defaults (from AlgorithmConfig).
        self.num_env_runners = 1

    @override(AlgorithmConfig)
    def training(self, *, num_episodes_per_train_batch=NotProvided, **kwargs) -> Self:
        """Sets the training related configuration.

        Args:
            num_episodes_per_train_batch: The number of complete episodes per train
                batch. VPG requires entire episodes to be sampled from the EnvRunners.
                For environments with varying episode lengths, this leads to varying
                batch sizes (in timesteps) as well possibly causing slight learning
                instabilities. However, for simplicity reasons, we stick to collecting
                always exactly n episodes per training update.

        Returns:
            This updated AlgorithmConfig object.
        """
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if num_episodes_per_train_batch is not NotProvided:
            self.num_episodes_per_train_batch = num_episodes_per_train_batch

        return self

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self):
        if self.framework_str == "torch":
            from ray.rllib.examples.rl_modules.classes.vpg_torch_rlm import (
                VPGTorchRLModule,
            )

            spec = RLModuleSpec(
                module_class=VPGTorchRLModule,
                model_config={"hidden_dim": 64},
            )
        else:
            raise ValueError(f"Unsupported framework: {self.framework_str}")

        return spec

    @override(AlgorithmConfig)
    def get_default_learner_class(self):
        if self.framework_str == "torch":
            from ray.rllib.examples.learners.utils.vpg_torch_learner import (
                VPGTorchLearner,
            )

            return VPGTorchLearner
        else:
            raise ValueError(f"Unsupported framework: {self.framework_str}")
