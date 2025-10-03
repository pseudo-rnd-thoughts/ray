"""An example script showing how to define and load an `RLModule` that applies
action masking

This example:
    - Defines an `RLModule` that applies action masking.
    - It does so by using a `gymnasium.spaces.dict.Dict` observation space
        with two keys, namely `"observations"`, holding the original observations
        and `"action_mask"` defining the action mask for the current environment
        state. Note, by this definition you can wrap any `gymnasium` environment
        and use it for this module.
    - Furthermore, it derives its `TorchRLModule` from the `PPOTorchRLModule` and
        can therefore be easily plugged into our `PPO` algorithm.
    - It overrides the `forward` methods of the `PPOTorchRLModule` to apply the
        action masking and it overrides the `_compute_values` method for GAE
        computation to extract the `"observations"` from the batch `Columns.OBS`
        key.
    - It uses the custom `ActionMaskEnv` that defines for each step a new action
        mask that defines actions that are allowed (1.0) and others that are not
        (0.0).
    - It runs 10 iterations with PPO and finishes.

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
You should expect a mean episode reward of around 0.35. The environment is a random
environment paying out random rewards - so the agent cannot learn, but it can obey the
action mask and should do so (no `AssertionError` should happen).
After 40,000 environment steps and 10 training iterations the run should stop
successfully:

+-------------------------------+------------+----------------------+--------+
| Trial name                    | status     | loc                  |   iter |
|                               |            |                      |        |
|-------------------------------+------------+----------------------+--------+
| PPO_ActionMaskEnv_dedc8_00000 | TERMINATED | 192.168.1.178:103298 |     10 |
+-------------------------------+------------+----------------------+--------+
+------------------+------------------------+------------------------+
|   total time (s) |   num_env_steps_sample |   num_env_steps_traine |
|                  |             d_lifetime |             d_lifetime |
+------------------+------------------------+------------------------+
|          57.9207 |                  40000 |                  40000 |
+------------------+------------------------+------------------------+
*------------------------+
|   num_episodes_lifetim |
|                      e |
+------------------------|
|                   3898 |
+------------------------+
"""

from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
from ray.rllib.examples.envs.utils.random_env import RandomEnv
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


class ActionMaskEnv(RandomEnv):
    """A randomly acting environment that publishes an action-mask each step."""

    def __init__(self, config):
        super().__init__(config)
        # Masking only works for Discrete actions.
        assert isinstance(self.action_space, Discrete)
        # Add action_mask to observations.
        self.observation_space = Dict(
            {
                "action_mask": Box(0.0, 1.0, shape=(self.action_space.n,)),
                "observations": self.observation_space,
            }
        )
        self.valid_actions = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset()
        self._fix_action_mask(obs)
        return obs, info

    def step(self, action):
        # Check whether action is valid.
        if not self.valid_actions[action]:
            raise ValueError(
                f"Invalid action ({action}) sent to env! "
                f"valid_actions={self.valid_actions}"
            )
        obs, rew, done, truncated, info = super().step(action)
        self._fix_action_mask(obs)
        return obs, rew, done, truncated, info

    def _fix_action_mask(self, obs):
        # Fix action-mask: Everything larger 0.5 is 1.0, everything else 0.0.
        self.valid_actions = np.round(obs["action_mask"])
        obs["action_mask"] = self.valid_actions


class ActionMaskingRLModule(RLModule):
    """An RLModule that implements an action masking for safe RL.

    This RLModule implements action masking to avoid unsafe/unwanted actions
    dependent on the current state (observations). It does so by using an
    environment generated action mask defining which actions are allowed and
    which should be avoided. The action mask is extracted from the
    environment's `gymnasium.spaces.dict.Dict` observation and applied after
    the module's `forward`-pass to the action logits. The resulting action
    logits prevent unsafe/unwanted actions to be sampled from the corresponding
    action distribution.

    Note, this RLModule is implemented for the `PPO` algorithm only. It is not
    guaranteed to work with other algorithms. Furthermore, not that for this
    module to work it requires an environment with a `gymnasium.spaces.dict.Dict`
    observation space containing tow key, `"action_mask"` and `"observations"`.
    """

    @override(RLModule)
    def __init__(
        self,
        *,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        inference_only: Optional[bool] = None,
        learner_only: bool = False,
        model_config: Optional[Union[dict, DefaultModelConfig]] = None,
        catalog_class=None,
        **kwargs,
    ):
        # If observation space is not of type `Dict` raise an error.
        if not isinstance(observation_space, gym.spaces.dict.Dict):
            raise ValueError(
                "This RLModule requires the environment to provide a "
                "`gym.spaces.Dict` observation space of the form: \n"
                " {'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                "  'observation_space': self.observation_space}"
            )

        # While the environment holds an observation space that contains, both,
        # the action mask and the original observation space, the 'RLModule'
        # receives only the `"observation"` element of the space, but not the
        # action mask.
        self.observation_space_with_mask = observation_space
        self.observation_space = observation_space["observations"]

        # Keeps track if observation specs have been checked already.
        self._checked_observations = False

        # The DefaultPPORLModule, in its constructor will build networks for the
        # original observation space (i.e. without the action mask).
        super().__init__(
            observation_space=self.observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )


class ActionMaskingTorchRLModule(ActionMaskingRLModule, PPOTorchRLModule):
    @override(PPOTorchRLModule)
    def setup(self):
        super().setup()
        # We need to reset here the observation space such that the
        # super`s (`PPOTorchRLModule`) observation space is the
        # original space (i.e. without the action mask) and `self`'s
        # observation space contains the action mask.
        self.observation_space = self.observation_space_with_mask

    @override(PPOTorchRLModule)
    def _forward_inference(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        # Preprocess the original batch to extract the action mask.
        action_mask, batch = self._preprocess_batch(batch)
        # Run the forward pass.
        outs = super()._forward_inference(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, action_mask)

    @override(PPOTorchRLModule)
    def _forward_exploration(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        # Preprocess the original batch to extract the action mask.
        action_mask, batch = self._preprocess_batch(batch)
        # Run the forward pass.
        outs = super()._forward_exploration(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, action_mask)

    @override(PPOTorchRLModule)
    def _forward_train(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        # Run the forward pass.
        outs = super()._forward_train(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, batch["action_mask"])

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, TensorType], embeddings=None):
        # Check, if the observations are still in `dict` form.
        if isinstance(batch[Columns.OBS], dict):
            # Preprocess the batch to extract the `observations` to `Columns.OBS`.
            action_mask, batch = self._preprocess_batch(batch)
            # NOTE: Because we manipulate the batch we need to add the `action_mask`
            # to the batch to access them in `_forward_train`.
            batch["action_mask"] = action_mask
        # Call the super's method to compute values for GAE.
        return super().compute_values(batch, embeddings)

    def _preprocess_batch(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Tuple[TensorType, Dict[str, TensorType]]:
        """Extracts observations and action mask from the batch

        Args:
            batch: A dictionary containing tensors (at least `Columns.OBS`)

        Returns:
            A tuple with the action mask tensor and the modified batch containing
                the original observations.
        """
        # Check observation specs for action mask and observation keys.
        self._check_batch(batch)

        # Extract the available actions tensor from the observation.
        action_mask = batch[Columns.OBS].pop("action_mask")

        # Modify the batch for the `DefaultPPORLModule`'s `forward` method, i.e.
        # pass only `"obs"` into the `forward` method.
        batch[Columns.OBS] = batch[Columns.OBS].pop("observations")

        # Return the extracted action mask and the modified batch.
        return action_mask, batch

    def _mask_action_logits(
        self, batch: Dict[str, TensorType], action_mask: TensorType
    ) -> Dict[str, TensorType]:
        """Masks the action logits for the output of `forward` methods

        Args:
            batch: A dictionary containing tensors (at least action logits).
            action_mask: A tensor containing the action mask for the current
                observations.

        Returns:
            A modified batch with masked action logits for the action distribution
            inputs.
        """
        # Convert action mask into an `[0.0][-inf]`-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)

        # Mask the logits.
        batch[Columns.ACTION_DIST_INPUTS] += inf_mask

        # Return the batch with the masked action logits.
        return batch

    def _check_batch(self, batch: Dict[str, TensorType]) -> Optional[ValueError]:
        """Assert that the batch includes action mask and observations.

        Args:
            batch: A dicitonary containing tensors (at least `Columns.OBS`) to be
                checked.

        Raises:
            `ValueError` if the column `Columns.OBS`  does not contain observations
                and action mask.
        """
        if not self._checked_observations:
            if "action_mask" not in batch[Columns.OBS]:
                raise ValueError(
                    "No action mask found in observation. This `RLModule` requires "
                    "the environment to provide observations that include an "
                    "action mask (i.e. an observation space of the Dict space "
                    "type that looks as follows: \n"
                    "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                    "'observations': self.observation_space}"
                )
            if "observations" not in batch[Columns.OBS]:
                raise ValueError(
                    "No observations found in observation. This 'RLModule` requires "
                    "the environment to provide observations that include the original "
                    "observations under a key `'observations'` in a dict (i.e. an "
                    "observation space of the Dict space type that looks as follows: \n"
                    "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                    "'observations': <observation_space>}"
                )
            self._checked_observations = True


parser = add_rllib_example_script_args(
    default_iters=10,
    default_timesteps=100000,
    default_reward=150.0,
)

args = parser.parse_args()

if args.algo != "PPO":
    raise ValueError("This example only supports PPO. Please use --algo=PPO.")

base_config = (
    PPOConfig()
    .environment(
        env=ActionMaskEnv,
        env_config={
            "action_space": Discrete(100),
            # This defines the 'original' observation space that is used in the
            # `RLModule`. The environment will wrap this space into a
            # `gym.spaces.Dict` together with an 'action_mask' that signals the
            # `RLModule` to adapt the action distribution inputs for the underlying
            # `DefaultPPORLModule`.
            "observation_space": Box(-1.0, 1.0, (5,)),
        },
    )
    .rl_module(
        # We need to explicitly specify here RLModule to use and
        # the catalog needed to build it.
        rl_module_spec=RLModuleSpec(
            module_class=ActionMaskingTorchRLModule,
            model_config={
                "head_fcnet_hiddens": [64, 64],
                "head_fcnet_activation": "relu",
            },
        ),
    )
    .evaluation(
        evaluation_num_env_runners=1,
        evaluation_interval=1,
        # Run evaluation parallel to training to speed up the example.
        evaluation_parallel_to_training=True,
    )
)

if __name__ == "__main__":
    # Run the example (with Tune).
    run_rllib_example_script_experiment(base_config, args)
