from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.examples.envs.utils.cartpole_with_dict_observation_space import (
    CartPoleWithDictObservationSpace,
)
from ray.rllib.examples.envs.utils.multi_agent.guess_the_number_game import (
    GuessTheNumberGame,
)
from ray.rllib.examples.envs.utils.multi_agent.two_step_game import (
    TwoStepGame,
    TwoStepGameWithGroupedAgents,
)
from ray.rllib.examples.envs.utils.nested_space_repeat_after_me_env import (
    NestedSpaceRepeatAfterMeEnv,
)
from ray.rllib.examples.envs.utils.stateless_cartpole import StatelessCartPole


# Backward compatibility.
__all__ = [
    "GuessTheNumberGame",
    "TwoStepGame",
    "TwoStepGameWithGroupedAgents",
]


MultiAgentCartPole = make_multi_agent("CartPole-v1")
MultiAgentMountainCar = make_multi_agent("MountainCarContinuous-v0")
MultiAgentPendulum = make_multi_agent("Pendulum-v1")
MultiAgentStatelessCartPole = make_multi_agent(lambda config: StatelessCartPole(config))
MultiAgentCartPoleWithDictObservationSpace = make_multi_agent(
    lambda config: CartPoleWithDictObservationSpace(config)
)
MultiAgentNestedSpaceRepeatAfterMeEnv = make_multi_agent(
    lambda config: NestedSpaceRepeatAfterMeEnv(config)
)
