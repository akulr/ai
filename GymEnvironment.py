import gym
import logging
from tqdm import tqdm
from statistics import mean


class GymEnvironment:

    def __init__(self, envName: str) -> None:
        self.envName = envName
        self.env = gym.make(envName)

    def getPossibleActions(self) -> list:
        action_space = self.env.action_space
        return list(range(action_space.start, action_space.n))

    def sample_action(self) -> int:
        return self.env.action_space.sample()
    
    def baselineExecution(self, iteration) -> None:
        reward_list = []
        step_list = []
        
        for i in tqdm(range(1, (iteration+1))):
            episode_step_count = 0
            episode_reward = 0

            self.env.reset()
            while True:
                action = self.env.action_space.sample()
                _, reward, _, _ = self.env.step(action)
                
                episode_step_count += 1
                episode_reward += reward

                if reward == 20:
                    logging.info(
                        f'Random Execution Iteration {i}; total Steps Taken {episode_step_count}; Rewards gained: {episode_reward}')
                    break
            reward_list.append(episode_reward)
            step_list.append(episode_step_count)
            
        print('=='*50)
        print('Random Execution')
        print('Mean Reward: {}'.format(mean(reward_list)))
        print('Mean Steps: {}'.format(mean(step_list)))
        print('=='*50)