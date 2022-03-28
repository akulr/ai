import gym
import logging

debug = False

logging.basicConfig(
    format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', 
    filename='gamePython.log', encoding='utf-8', level=(logging.DEBUG if debug else logging.INFO)
)

gym_environment = 'Taxi-v3'
env = gym.make(gym_environment)
logging.info('{} GYM environment initiated'.format(gym_environment))

game_count = 20
max_steps = 10000000000
logging.info('{} Games played with max steps are {}'.format(
    game_count, max_steps))


for i_episode in range(game_count):
    observation = env.reset()  # Reset the environment state

    episode_reward = 0
    for t in range(max_steps):
        env.render()

        # print(observation)
        # Sample actions set
        action = env.action_space.sample()

        # Result of the action performed
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        
        logging.debug('{} action: {} and achived Reward is {}'.format(
            (t+1), action, reward))

        # check if the  game's environment is completed
        if reward == 20:
            logging.info("Episode {} finished after {} steps with episode rewards: {}".format(
                (i_episode+1), (t+1), episode_reward))
            break
env.close()
