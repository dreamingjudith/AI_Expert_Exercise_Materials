import time

import gym
import envs
import numpy as np

np.set_printoptions(precision=3, suppress=True, threshold=10000, linewidth=250)

""" Load environment """
# env_name = 'MazeSample5x5-v0'
# env_name = 'MazeSample10x10-v0'
# env_name = 'MazeRandom10x10-v0'
# env_name = 'MazeRandom10x10-plus-v0'
# env_name = 'MazeRandom20x20-v0'
# env_name = 'MazeRandom20x20-plus-v0'
env_name = 'MyCartPole-v0'
# env_name = 'MyMountainCar-v0'

env = gym.make(env_name)

"""
env.S: the number of states (integer)
env.A: the number of actions (integer)
env.T: transition matrix (S x A x S)-sized array
env.R: reward matrix (S x A)-sized array
env.gamma: discount factor (0 ~ 1)
"""


def policy_evaluation(env, pi):
    """
    :param env: MDP(S, A, T, R, gamma)
    :param pi: behavior policy (S x A)-sized array
    :return: V, Q where V is (S)-sized array and Q is (S x A)-sized array
    """
    r = np.sum(env.R * pi, axis=1)
    P = np.tensordot(pi, env.T, axes=([1], [1]))[np.arange(env.S), np.arange(env.S), :]
    V = np.linalg.inv(np.eye(env.S) - env.gamma * P).dot(r)
    Q = env.R + env.gamma * env.T.dot(V)

    return V, Q


def policy_improvement(env, Q):
    ###################
    # TODO: 주어진 Q에 대해 향상된 정책(greedy policy)를 계산하는 코드를 작성하세요.
    # ...
    ###################
    pi = np.zeros((env.S, env.A))
    for s in range(env.S):
        best_action_idx = np.argmax(Q[s])
        pi[s][best_action_idx] = 1.

    return pi


def policy_iteration(env):
    pi = np.ones((env.S, env.A)) / env.A
    Q = np.zeros((env.S, env.A))
    for i in range(1000):
        ###################
        # TODO: 여기를 작성하세요.
        # ...
        V, Q = policy_evaluation(env, pi)
        new_pi = policy_improvement(env, Q)
        ###################
        if np.all(pi == new_pi):
            break
        pi = new_pi
    return pi, Q


def value_iteration(env):
    ###################
    # TODO: 여기를 작성하세요.
    Q = np.zeros((env.S, env.A))
    for i in range(1000):
        Q = np.zeros((env.S, env.A))
    pi = np.ones((env.S, env.A)) / env.A
    ###################
    return pi, Q

pi, Q = policy_iteration(env)
pi = np.ones((env.S, env.A)) / env.A  # Uniform distribution

for episode in range(10):
    state = env.reset()
    env.render()

    episode_reward = 0.
    for t in range(10000):
        action = int(np.random.choice(np.arange(env.A), p=pi[state, :]))
        state1, reward, done, info = env.step(action)
        episode_reward += reward
        print("[%4d] state=%4s / action=%d / reward=%7.4f / state1=%4s / info=%s" % (t, state, action, reward, state1, info))

        # env.draw_policy_evaluation(Q, pi)  # 필요시 주석 처리
        env.render()
        time.sleep(0.3 if 'Maze' in env_name else 0.01)

        if done:
            break
        state = state1
    print('Episode reward: %.4f' % episode_reward)

    time.sleep(1)
time.sleep(10)
