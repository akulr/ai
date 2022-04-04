import gym
import tf_agents

def transformObservation(env):
    def decodeState(state_int):
        return list(env.decode(state_int))
    return gym.wrappers.TransformObservation(env, decodeState)

env = tf_agents.environments.suite_gym.load(
    environment_name="Taxi-v3",
    discount=1,
    gym_env_wrappers=(transformObservation)
)