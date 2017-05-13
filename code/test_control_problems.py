# implementation based on https://github.com/lisa-1010/transfer_rl/blob/master/code/gym_pipeline.py

import gym
import matplotlib.pyplot as plt

# TODO: import agents

ENV_NAME = 'CartPole-v0'
NUM_TEST_TRIALS = 100


class Pipeline(object):
    def __init__(self, env_name='InvertedPendulum-v1'):
        self.env = gym.make(env_name)
        # get action + observation space, pass into agent
        # TODO: define Agent (e.g. DQN or MCTS + DKT)
        # self.agent =
        self.test_performances = []
        # self.env.monitor.start('../experiments/' + env_name)

    def run_episode(self):
        state = self.env.reset()
        for timestep in xrange(self.env.spec.timestep_limit):
            action = self.agent.get_noisy_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.agent.perceive_and_train(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

    def run_test(self, episode_num):
        # run current agent model on environment,
        # evaluate average reward, create video
        total_reward = 0.0
        for episode in xrange(NUM_TEST_TRIALS):
            state = self.env.reset()
            for step in xrange(self.env.spec.timestep_limit):
                # self.env.render()
                action = self.agent.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                total_reward += reward
                if done:
                    break
        avg_reward = total_reward / NUM_TEST_TRIALS
        self.test_performances.append(avg_reward)
        print 'Episode: {} Average Reward Per Episode : {} '.format(episode_num,avg_reward)

    def run(self, num_episodes):
        for episode in xrange(num_episodes):
            self.run_episode()
            # Every 100 episodes, run test and print average reward
            if episode % 100 == 0:
                self.run_test(episode)
        self.env.monitor.close()

    def plot_results(self):
        plt.xlabel("Episode")
        plt.ylabel("Average Test Reward")
        plt.plot(self.test_performances)
        plt.savefig('Reward-Vs-Episode')


if __name__ == "__main__":
    pipeline = Pipeline(env_name=ENV_NAME)
    pipeline.run(num_episodes=100000)
    pipeline.plot_results()


