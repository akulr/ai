from GymEnvironment import GymEnvironment
from tqdm import tqdm
import logging
import util
import pickle   # nosec
import os


class ReinforcementLearning:

    def __init__(self, env: GymEnvironment, alpha: float = 1.0, epsilon: float = 0.05, gamma: float = 0.8, model_file: str = None) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._q_val = util.Counter()
        self.model_file = 'ModelOutput/RL-Qlearning.pkl' if model_file is None else model_file
        self.load_model()

    def getAction(self, state) -> int:
        if util.flipCoin(self.epsilon):
            return self.env.sample_action()
        return self.getPolicy(state)

    def updateQvalues(self, state: int, action: int, reward: float, nextState: int) -> None:
        sample = self.alpha * \
            (reward + (self.gamma * self.computeStateQvalue(nextState)))
        self._q_val[(nextState, action)] = ((1 - self.alpha) *
                                            self._q_val[(state, action)]) + sample

    def getPolicy(self, state: int) -> int:
        action_cntr = util.Counter()
        for action in self.env.getPossibleActions():
            action_cntr[action] = self._q_val[(state, action)]
        return action_cntr.argMax()

    def computeStateQvalue(self, state: int) -> float:
        return max([self._q_val[(state, action)] for action in self.env.getPossibleActions()])

    def start_learning(self, iteration_count: int, saveAfter: int = 1) -> None:
        iteration_count += 1
        for i in tqdm(range(1, iteration_count)):
            episode_reward = 0
            step = 0

            observation = self.env.env.reset()
            while True:
                # self.env.env.render()

                action = self.getAction(observation)
                observation, reward, done, info = self.env.env.step(action)

                episode_reward += reward
                step += 1

                if (step % 500) == 0:
                    logging.debug(
                        f'Training Iteration {i}; Total Steps {step}; Reward gained: {episode_reward}; Action: {action}; status: {done}')

                if reward == 20:
                    logging.info(
                        f'Training Iteration {i}; total Steps Taken {step}; Rewards gained: {episode_reward}')
                    break

            if (i % saveAfter) == 0:
                logging.info(f'Training Iteration {i}; Creating pickle dump of the Q-values')
                self.store_model()

    def store_model(self) -> None:
        folder_path = os.path.dirname(self.model_file)
        if len(folder_path) != 0:
            os.makedirs(folder_path, exist_ok=True)
        pickle.dump(self._q_val, open(self.model_file, 'wb'))

    def load_model(self) -> None:
        if not os.path.isfile(self.model_file):
            logging.info('No model created')
            return None
        self._q_val = pickle.load(open(self.model_file, 'rb'))    # nosec
