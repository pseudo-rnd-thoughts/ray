import tree  # pip install dm_tree

from ray.rllib.algorithms import Algorithm
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    ENV_RUNNER_SAMPLING_TIMER,
    LEARNER_RESULTS,
    LEARNER_UPDATE_TIMER,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TIMERS,
)
from rllib.examples.algorithms.custom.utils.vcp_config import VPGConfig


class VPG(Algorithm):
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> VPGConfig:
        return VPGConfig()

    @override(Algorithm)
    def training_step(self) -> None:
        """Override of the training_step method of `Algorithm`.

        Runs the following steps per call:
        - Sample B timesteps (B=train batch size). Note that we don't sample complete
        episodes due to simplicity. For an actual VPG algo, due to the loss computation,
        you should always sample only completed episodes.
        - Send the collected episodes to the VPG LearnerGroup for model updating.
        - Sync the weights from LearnerGroup to all EnvRunners.
        """
        # Sample.
        with self.metrics.log_time((TIMERS, ENV_RUNNER_SAMPLING_TIMER)):
            episodes, env_runner_results = self._sample_episodes()
        # Merge results from n parallel sample calls into self's metrics logger.
        self.metrics.aggregate(env_runner_results, key=ENV_RUNNER_RESULTS)

        # Just for demonstration purposes, log the number of time steps sampled in this
        # `training_step` round.
        # Mean over a window of 100:
        self.metrics.log_value(
            "episode_timesteps_sampled_mean_win100",
            sum(map(len, episodes)),
            reduce="mean",
            window=100,
        )
        # Exponential Moving Average (EMA) with coeff=0.1:
        self.metrics.log_value(
            "episode_timesteps_sampled_ema",
            sum(map(len, episodes)),
            ema_coeff=0.1,  # <- weight of new value; weight of old avg=1.0-ema_coeff
        )

        # Update model.
        with self.metrics.log_time((TIMERS, LEARNER_UPDATE_TIMER)):
            learner_results = self.learner_group.update(
                episodes=episodes,
                timesteps={
                    NUM_ENV_STEPS_SAMPLED_LIFETIME: (
                        self.metrics.peek(
                            (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME)
                        )
                    ),
                },
            )
        # Merge results from m parallel update calls into self's metrics logger.
        self.metrics.aggregate(learner_results, key=LEARNER_RESULTS)

        # Sync weights.
        with self.metrics.log_time((TIMERS, SYNCH_WORKER_WEIGHTS_TIMER)):
            self.env_runner_group.sync_weights(
                from_worker_or_learner_group=self.learner_group,
                inference_only=True,
            )

    def _sample_episodes(self):
        # How many episodes to sample from each EnvRunner?
        num_episodes_per_env_runner = self.config.num_episodes_per_train_batch // (
            self.config.num_env_runners or 1
        )
        # Send parallel remote requests to sample and get the metrics.
        sampled_data = self.env_runner_group.foreach_env_runner(
            # Return tuple of [episodes], [metrics] from each EnvRunner.
            lambda env_runner: (
                env_runner.sample(num_episodes=num_episodes_per_env_runner),
                env_runner.get_metrics(),
            ),
            # Loop over remote EnvRunners' `sample()` method in parallel or use the
            # local EnvRunner if there aren't any remote ones.
            local_env_runner=self.env_runner_group.num_remote_workers() <= 0,
        )
        # Return one list of episodes and a list of metrics dicts (one per EnvRunner).
        episodes = tree.flatten([s[0] for s in sampled_data])
        stats_dicts = [s[1] for s in sampled_data]

        return episodes, stats_dicts
