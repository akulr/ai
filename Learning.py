from email import parser
import logging
import argparse

from GymEnvironment import GymEnvironment
from ReinforcementLearning_Qvalue import ReinforcementLearning_Qvalue

parser = argparse.ArgumentParser()

parser.add_argument("--debug", action="store_true", help="Execute in Debug mode")
parser.add_argument("--gym-env", default='Taxi-v3', help="Gym environment")
parser.add_argument("-a", "--alpha", type=float, default=0.9, help="Qvalue alpha")
parser.add_argument("-d", "--discount", type=float, default=0.2, help="Qvalue discount value")
parser.add_argument("--ep", type=float, default=0.5, help="Qvalue epsilon")
parser.add_argument("--q-model", type=int, default=None, help="Model filename")
parser.add_argument("--itr-training", type=int, default=5000, help="Total iteration for learning")
parser.add_argument("--itr-testing", type=int, default=100, help="Total iteration for testing")

args = parser.parse_args()

print(f'Argument: {args}')

env_name = 'Taxi-v3'
GYM_ENV = GymEnvironment(envName=args.gym_env)

logging.basicConfig(
    format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    filename='gamePython.log', encoding='utf-8', level=(logging.DEBUG if args.debug else logging.INFO)
)

logging.info(f'Executing with Parameters: {args}')

def environment_init():
    # env_name = 'Taxi-v3'
    # GYM_ENV = GymEnvironment(envName=args.gym_env)
    logging.info(f'Loaded gym environmnet {env_name}')
    GYM_ENV.baselineExecution(iteration=args.itr_testing)

def qvalue():
    rl = ReinforcementLearning_Qvalue(
        GYM_ENV,
        alpha=args.alpha,
        epsilon=args.ep, 
        gamma=args.discount,
        model_file=args.q_model
    )
    rl.start_learning(iteration_count=args.itr_training, saveAfter=10)
    rl.test_execution(iteration_count=args.itr_testing)

if __name__ == "__main__":
    environment_init()
    qvalue()
    pass