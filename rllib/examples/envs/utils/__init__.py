from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.examples.envs.utils.stateless_cartpole import StatelessCartPole

MultiAgentCartPole = make_multi_agent("CartPole-v1")
MultiAgentPendulum = make_multi_agent("Pendulum-v1")
MultiAgentStatelessCartPole = make_multi_agent(lambda config: StatelessCartPole(config))
