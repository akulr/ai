from tqdm import tqdm
import tensorflow as tf
import tf_agents
import logging
import pickle   # nosec
import os

from GymEnvironment import GymEnvironment

# num_iterations = 20000 # @param {type:"integer"}

# initial_collect_steps = 100  # @param {type:"integer"}
# collect_steps_per_iteration =   1# @param {type:"integer"}
# replay_buffer_max_length = 100000  # @param {type:"integer"}

# batch_size = 64  # @param {type:"integer"}
# learning_rate = 1e-3  # @param {type:"number"}
# log_interval = 200  # @param {type:"integer"}

# num_eval_episodes = 10  # @param {type:"integer"}
# eval_interval = 1000  # @param {type:"integer"}


class ReinforcementLearning_DQN:
    def __init__(self,
                 env: GymEnvironment,
                 learning_rate: float = 1e-3,
                 replay_buffer_capacity: int = 100000,
                 model_file: str = None,
                 ) -> None:
        # Hyperparameters
        self.env = env
        self.learning_rate = learning_rate
        self.replay_buffer_capacity = replay_buffer_capacity
        self.model_file = 'ModelOutput/RL-Qlearning.pkl' if model_file is None else model_file
        self.initialize_environment()
        self.load_model()

    def store_model(self) -> None:
        folder_path = os.path.dirname(self.model_file)
        if len(folder_path) != 0:
            os.makedirs(folder_path, exist_ok=True)
        pickle.dump(self.agent, open(self.model_file, 'wb'))

    def load_model(self) -> None:
        if not os.path.isfile(self.model_file):
            logging.info('No model created')
            return None
        self.agent = pickle.load(open(self.model_file, 'rb'))    # nosec

    def initialize_environment(self):
        # GYM enviroment
        self.train_env = self.env.tf_environment()
        self.eval_env = self.env.tf_environment()

        # Agent
        self.agent = self.create_agent()
        self.agent.initialize()

        # Replay Buffer
        self.replay_buffer = tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_capacity
        )

    def create_agent(self):
        fc_layer_params = (100, 50)

        # Dense layers
        dense_layers = [
            tf.keras.layers.Dense(
                num_units, activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2.0, mode='fan_in', distribution='truncated_normal')
            )
            for num_units in fc_layer_params
        ]

        # Q value Layer
        q_values_layer = tf.keras.layers.Dense(
            self.env.action_size, activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2)
        )

        model_layer = dense_layers + [q_values_layer]

        # Qlearning network
        qlearning_network = tf_agents.networks.sequential.Sequential(model_layer)

        # DQN Agent
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        train_step_counter = tf.Variable(0)
        return tf_agents.agents.dqn.dqn_agent.DqnAgent(
            self.train_env.time_step_spec(), self.train_env.action_spec(), q_network=qlearning_network, optimizer=optimizer,
            td_errors_loss_fn=tf_agents.utils.common.element_wise_squared_loss, train_step_counter=train_step_counter
        )

    def compute_avg_return(self, environment, policy, num_episodes=10, islogging=False):
        total_return = 0.0
        for i in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0
            episode_step = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
                episode_step += 1

            total_return += episode_return
            if islogging:
                logging.info(
                    f'Random Execution Iteration {(i+1)}; total Steps Taken {episode_step}; Rewards gained: {episode_return}')

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def create_driver(self, policy, steps):
        return tf_agents.drivers.py_driver.PyDriver(
            env=self.env.env,
            observers=[self.replay_buffer.add_batch],
            policy=policy,
            max_steps=steps
        )

    def train_agent(self,
                    initial_collect_steps=100, batch_size=64, num_eval_episodes=10,
                    collect_steps_per_iteration=1, num_iterations=20000,
                    log_interval=200, eval_interval=1000
                    ):
        # Filling buffer with environment observation samples
        self.create_driver(
            tf_agents.policies.py_tf_eager_policy.PyTFEagerPolicy(
                tf_agents.policies.random_tf_policy.RandomTFPolicy(
                    self.train_env.time_step_spec(), self.train_env.action_spec()
                ), use_tf_function=True
            ), steps=initial_collect_steps
        ).run(self.train_env.reset())

        # creating an iterator from the buffer of size batch_size
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2
        ).prefetch(3)
        iterator = iter(dataset)

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        self.agent.train = tf_agents.utils.common.function(self.agent.train)
        # Reset the train step.
        self.agent.train_step_counter.assign(0)
        # Evaluate the agent's policy once before training.
        avg_return = self.compute_avg_return(
            self.eval_env, self.agent.policy, num_eval_episodes)
        returns = [avg_return]

        time_step = self.train_env.reset()

        collect_driver = self.create_driver(
            tf_agents.policies.py_tf_eager_policy.PyTFEagerPolicy(
                self.agent.collect_policy, use_tf_function=True
            ), collect_steps_per_iteration
        )

        for _ in tqdm(range(num_iterations)):
            # Collect a few steps and save to the replay buffer.
            time_step, _ = collect_driver.run(time_step)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, _ = next(iterator)
            train_loss = self.agent.train(experience)

            # Log the progress
            step = self.agent.train_step_counter.numpy()
            if step % log_interval == 0:
                logging.info(
                    f'DQN Training; Step: {step}; Training Loss: {train_loss}')
            if step % eval_interval == 0:
                avg_loss = self.compute_avg_return(
                    self.eval_env, self.agent.policy, num_eval_episodes)
                logging.info(
                    f'DQN Training; Step: {step}; Evaluation Loss: {train_loss}')
                returns.append(avg_loss)
                self.store_model()
        pass
