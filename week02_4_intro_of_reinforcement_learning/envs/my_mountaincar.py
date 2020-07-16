"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import os


class MyMountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high)

        # discretize the state space (added by jmlee)
        self.N_POSITION = 15
        self.N_VELOCITY = 15
        self.position_slices = np.linspace(self.min_position, self.max_position, self.N_POSITION)
        self.velocity_slices = np.linspace(-self.max_speed, self.max_speed, self.N_VELOCITY)
        self._discretize()

        self.seed()
        self.reset()

    def _discretize(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(dir_path, "env_samples", 'mountaincar_%d_%d.npy' % (self.N_POSITION, self.N_VELOCITY))

        if os.path.exists(file_path):
            mdp = np.load(file_path, allow_pickle=True)[()]
            self.S, self.A, self.T, self.R, self.gamma = mdp['S'], mdp['A'], mdp['T'], mdp['R'], mdp['gamma']
        else:
            # (position, velocity) -> (N_POSITION x N_VELOCITY)
            print('MountainCar environment not found: creating...')
            self.S = self.N_POSITION * self.N_VELOCITY + 1
            self.A = 3
            self.T = np.zeros((self.S, self.A, self.S))
            self.R = np.zeros((self.S, self.A))
            self.T[self.S - 1, :, self.S - 1] = 1.
            self.gamma = 0.99
            for state in range(self.S - 1):
                for action in range(self.A):
                    for i in range(1000):
                        self.reset()
                        ob = np.array(self.sample_ob_from_state(state))
                        self.state = np.array(ob)
                        state1, reward, done, ob1 = self.step(action)
                        self.R[state, action] += reward
                        ob1 = np.array(ob1)
                        # print("[%4d] state=%s (%3d) / action=%s / reward=%f / state1=%s (%3d) / done=%s" % (i, ob, state, action, reward, ob1, state1, done))
                        if done:
                            self.T[state, action, self.S - 1] += 1
                        else:
                            self.T[state, action, state1] += 1
                    self.T[state, action, :] /= np.sum(self.T[state, action, :])
                    self.R[state, action] /= 1000
                    # print(self.T[state, action, :])
            np.save(file_path, {'S': self.S, 'A': self.A, 'T': self.T, 'R': self.R, 'gamma': self.gamma})
            print('Finished!')

    def ob_to_state(self, ob):
        position, velocity = ob
        position_idx = np.where(self.position_slices <= position)[0][-1]
        velocity_idx = np.where(self.velocity_slices <= velocity)[0][-1]

        return position_idx * self.N_VELOCITY + velocity_idx

    def sample_ob_from_state(self, state):
        position_idx_low = state // self.N_VELOCITY
        position_idx_high = np.min([position_idx_low + 1, self.N_POSITION - 1])
        velocity_idx_low = state % self.N_VELOCITY
        velocity_idx_high = np.min([velocity_idx_low + 1, self.N_VELOCITY - 1])

        position = np.random.uniform(self.position_slices[position_idx_low], self.position_slices[position_idx_high])
        velocity = np.random.uniform(self.velocity_slices[velocity_idx_low], self.velocity_slices[velocity_idx_high])

        return (position, velocity)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action-1)*0.001 + math.cos(3*position)*(-0.0025)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position)
        if done:
            reward = 100
        else:
            reward = 0

        self.state = (position, velocity)
        info = {'state': np.array(self.state)}
        return self.ob_to_state(self.state), reward, done, info
        # return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return self.ob_to_state(self.state)
        # return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def draw_policy_evaluation(self, Q, pi=None):
        pass
