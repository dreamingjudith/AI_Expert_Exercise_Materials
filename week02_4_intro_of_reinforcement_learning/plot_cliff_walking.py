import time
import matplotlib.pyplot as plt
import gym
import envs
import numpy as np
from exercise_3 import update_sarsa, update_q_learning, epsilon_greedy

np.set_printoptions(precision=3, suppress=True, threshold=10000, linewidth=250)


def epsilon_greedy(num_actions, Q, s, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(0, num_actions)
    else:
        return np.random.choice(np.flatnonzero(Q[s, :] == Q[s, :].max()))

def run():
    env_name = 'MyCliffWalking-v0'
    env = gym.make(env_name)
    env.T = env.R = None

    step_size = 0.2

    sarsa_Q = np.zeros((env.S, env.A))
    qlear_Q = np.zeros((env.S, env.A))
    epsilon = 0.2

    train_sarsa_eprs, train_qlear_eprs = [], []
    eval_sarsa_eprs, eval_qlear_eprs = [], []
    for episode in range(1000):
        state = env.reset()

        # train SARSA
        epr = 0.
        for t in range(10000):
            action = epsilon_greedy(env.A, sarsa_Q, state, epsilon)
            next_state, reward, done, info = env.step(action)
            next_action = epsilon_greedy(env.A, sarsa_Q, next_state, epsilon)

            sarsa_Q[state, action] = update_sarsa(step_size, sarsa_Q, state, action, reward, done, next_state, next_action, env.gamma)
            epr += reward
            if done:
                break
            state = next_state
        train_sarsa_eprs.append(epr)

        # train Q-learning
        state = env.reset()
        epr = 0.
        for t in range(10000):
            action = epsilon_greedy(env.A, qlear_Q, state, epsilon)
            next_state, reward, done, info = env.step(action)

            qlear_Q[state, action] = update_q_learning(step_size, qlear_Q, state, action, reward, done, next_state, env.gamma)
            epr += reward
            if done:
                break
            state = next_state
        train_qlear_eprs.append(epr)
        
        # evaluate SARSA
        state = env.reset()
        epr = 0.
        for t in range(10000):
            action = epsilon_greedy(env.A, sarsa_Q, state, 0)
            state, reward, done, _ = env.step(action)
            epr += reward
            if done:
                break
        eval_sarsa_eprs.append(epr)
        
        # evaluate Q-learning
        state = env.reset()
        epr = 0.
        for t in range(10000):
            action = epsilon_greedy(env.A, qlear_Q, state, 0)
            state, reward, done, _ = env.step(action)
            epr += reward
            if done:
                break
        eval_qlear_eprs.append(epr)

        print('[%4d] SARSA Episode reward=%.4f / Q-learning Episode reward=%.4f ' % (episode, train_sarsa_eprs[-1], train_qlear_eprs[-1]))

    def smoother(data):
        return [np.mean(data[i-50:i]) for i in range(50, 1000)]

    plt.plot(smoother(train_sarsa_eprs), label='SARSA (train)', color='red', linestyle=':')
    plt.plot(smoother(train_qlear_eprs), label='Q-learning (train)', color='blue', linestyle=':')
    plt.plot(smoother(eval_sarsa_eprs), label='SARSA (eval)', color='red')
    plt.plot(smoother(eval_qlear_eprs), label='Q-learning (eval)', color='blue')
    plt.legend()
    plt.title('Cliff Walking Example')
    plt.ylabel('Reward per episode')
    plt.xlabel('Episodes')
    plt.show()

if __name__ == "__main__":
    run()