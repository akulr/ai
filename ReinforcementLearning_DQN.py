import tensorflow as tf
import tf_agents
import logging

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
    def __init__(
        self,
        env: GymEnvironment,
        learning_rate: float,
        replay_buffer_capacity=100000
    ) -> None:
        # Hyperparameters
        self.env = env
        self.learning_rate = learning_rate
        self.replay_buffer_capacity = replay_buffer_capacity
        pass

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
        dense_layers.append(tf.keras.layers.Dense(
            self.env.action_size, activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2)
        ))

        # Qlearning network
        qlearning_network = tf_agents.networks.sequential.Sequence(
            dense_layers)

        # DQN Agent
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        train_step_counter = tf.Variable(0)
        return tf_agents.agents.dqn.dqn_agent.DqnAgent(
            self.train_env.time_step_spec(), self.train_env.action_spec(), q_network=qlearning_network, optimizer=optimizer,
            td_errors_loss_fn=tf_agents.utils.common.element_wise_squared_loss, train_step_counter=train_step_counter
        )

    def compute_avg_return(self, environment, policy, num_episodes=10):
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
            logging.info(f'Random Execution Iteration {(i+1)}; total Steps Taken {episode_step}; Rewards gained: {episode_return}')

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def create_driver(self, policy, steps):
        return tf_agents.drivers.py_driver.PyDriver(
            env=self.env.env,
            observers=[self.replay_buffer.add_batch],
            policy=policy,
            max_steps=steps
        )

    def train_agent(self):

        pass

    def eval_agent(self):
        
        pass
