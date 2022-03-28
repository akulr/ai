import gym


class GymEnvironment:

    def __init__(self, envName: str) -> None:
        self.envName = envName
        self.env = gym.make(envName)

    def getPossibleActions(self) -> list:
        action_space = self.env.action_space
        return list(range(action_space.start, action_space.n))

    def sample_action(self) -> int:
        return self.env.action_space.sample()
