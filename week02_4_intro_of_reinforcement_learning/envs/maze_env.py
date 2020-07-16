import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from envs.maze_view_2d import MazeView2D
import pygame

class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file=None, maze_size=None, mode=None):

        self.viewer = None

        if maze_file:
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640))
        elif maze_size:
            if mode == "plus":
                has_loops = True
                num_portals = int(round(min(maze_size)/3))
            else:
                has_loops = False
                num_portals = 0

            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size, screen_size=(640, 640),
                                        has_loops=has_loops, num_portals=num_portals)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high =  np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # For MDP planning (added by jmlee)
        self.S = self.maze_size[0] * self.maze_size[1] + 1  # 25 + 1 (including absorbing state)
        self.A = 4
        self.T = np.zeros((self.S, self.A, self.S))
        self.R = np.zeros((self.S, self.A))
        self.T[self.S-1, :, self.S-1] = 1.
        for s in range(self.S - 1):
            for a in range(self.A):
                self.maze_view.robot = self.state_to_ob(s)
                _, reward, done, ob1 = self.step(a)
                self.reset()
                # print("S=%s (%d) / A=%s / S1=%s (%d) / R=%f / done=%s" % (
                #     self.state_to_ob(s), s, self.ACTION[a], state1, self.ob_to_state(state1), reward, done))

                if done:
                    self.T[s, a, self.S - 1] = 1.
                else:
                    self.T[s, a, self.ob_to_state(ob1)] = 1.
                self.R[s, a] = reward
        self.gamma = 0.999

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self._configure()

    def ob_to_state(self, ob):
        return int(ob[1] * self.maze_size[1] + ob[0])

    def state_to_ob(self, state):
        return np.array([state % self.maze_size[1], state // self.maze_size[1]], dtype=int)

    def __del__(self):
        self.maze_view.quit_game()

    def _configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if isinstance(action, int) or isinstance(action, np.int64) or isinstance(action, np.int32):
            self.maze_view.move_robot(self.ACTION[action])
        else:
            self.maze_view.move_robot(action)

        if np.array_equal(self.maze_view.robot, self.maze_view.goal):
            reward = 1
            done = True
        else:
            reward = -0.1/(self.maze_size[0]*self.maze_size[1])
            done = False

        self.state = self.maze_view.robot

        info = self.state

        return self.ob_to_state(self.state), reward, done, info

    def reset(self):
        self.maze_view.reset_robot()
        self.state = np.zeros(2)
        self.steps_beyond_done = None
        self.done = False
        return self.ob_to_state(self.state)

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)

    def draw_policy_evaluation(self, Q, pi=None):
        self.maze_view.Q = Q
        if pi is None:
            self.maze_view.pi = np.zeros((self.S, self.A))
        else:
            self.maze_view.pi = pi
        self.maze_view.is_policy_evaluation = True
        self.maze_view.update()


class MazeEnvSample5x5(MazeEnv):

    def __init__(self):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy")


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom5x5, self).__init__(maze_size=(5, 5))


class MazeEnvSample10x10(MazeEnv):

    def __init__(self):
        super(MazeEnvSample10x10, self).__init__(maze_file="maze2d_10x10.npy")


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom10x10, self).__init__(maze_size=(10, 10))


class MazeEnvSample3x3(MazeEnv):

    def __init__(self):
        super(MazeEnvSample3x3, self).__init__(maze_file="maze2d_3x3.npy")


class MazeEnvRandom3x3(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom3x3, self).__init__(maze_size=(3, 3))


class MazeEnvSample100x100(MazeEnv):

    def __init__(self):
        super(MazeEnvSample100x100, self).__init__(maze_file="maze2d_100x100.npy")


class MazeEnvRandom100x100(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom100x100, self).__init__(maze_size=(100, 100))


class MazeEnvRandom10x10Plus(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom10x10Plus, self).__init__(maze_size=(10, 10), mode="plus")


class MazeEnvRandom20x20Plus(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom20x20Plus, self).__init__(maze_size=(20, 20), mode="plus")


class MazeEnvRandom30x30Plus(MazeEnv):
    def __init__(self):
        super(MazeEnvRandom30x30Plus, self).__init__(maze_size=(30, 30), mode="plus")
