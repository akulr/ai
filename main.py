import logging

from GymEnvironment import GymEnvironment
from ReinforcementLearning import ReinforcementLearning

debug = False

logging.basicConfig(
    format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    filename='gamePython.log', encoding='utf-8', level=(logging.DEBUG if debug else logging.INFO)
)

env_name = 'Taxi-v3'
env = GymEnvironment(envName=env_name)
logging.info(f'Loaded gym environmnet {env_name}')

rl = ReinforcementLearning(
    env, alpha=0.9, epsilon=0.5, gamma=0.8, model_file=None)
rl.start_learning(iteration_count=50000, saveAfter=100)

rl.test_execution(iteration_count=20)
