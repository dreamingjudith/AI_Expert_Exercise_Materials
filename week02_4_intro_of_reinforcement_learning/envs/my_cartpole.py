"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import logging
import math
import os

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

logger = logging.getLogger(__name__)


class MyCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        self.high = np.array([
            self.x_threshold * 2,
            3,
            self.theta_threshold_radians * 2,
            3])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-self.high, self.high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        # discretize the state space (added by jmlee)
        self.N_X = 6
        self.N_X_DOT = 3
        self.N_THETA = 9
        self.N_THETA_DOT = 4
        self.x_slices = np.r_[-self.high[0], np.linspace(-self.x_threshold, self.x_threshold, self.N_X - 2), self.high[0]]
        self.x_dot_slices = np.linspace(-self.high[1], self.high[1], self.N_X_DOT)
        self.theta_slices = np.r_[-self.high[2], np.linspace(-self.theta_threshold_radians, self.theta_threshold_radians, self.N_THETA - 2), self.high[2]]
        self.theta_dot_slices = np.linspace(-self.high[3], self.high[3], self.N_THETA_DOT)
        self._discretize()

    def _discretize(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(dir_path, "env_samples", 'cartpole_%d_%d_%d_%d.npy' % (self.N_X, self.N_X_DOT, self.N_THETA, self.N_THETA_DOT))

        if os.path.exists(file_path):
            mdp = np.load(file_path, allow_pickle=True)[()]
            self.S, self.A, self.T, self.R, self.gamma = mdp['S'], mdp['A'], mdp['T'], mdp['R'], mdp['gamma']
        else:
            print('CartPole environment not found: creating...')
            self.S = self.N_X * self.N_X_DOT * self.N_THETA * self.N_THETA_DOT + 1
            self.A = 2
            self.T = np.zeros((self.S, self.A, self.S))
            self.R = np.zeros((self.S, self.A))
            self.T[self.S - 1, :, self.S - 1] = 1.
            self.gamma = 0.99
            self._reset()
            for state in range(self.S - 1):
                for action in range(self.A):
                    for i in range(1000):
                        ob = np.array(self.sample_ob_from_state(state))
                        self.state = np.array(ob)
                        state1, reward, done, ob1 = self.step(action)
                        ob1 = np.array(ob1)
                        print("[%4d] state=%s (%3d) / action=%s / reward=%f / state1=%s (%3d) / done=%s" % (i, ob, state, action, reward, ob1, state1, done))
                        if done:
                            self.T[state, action, self.S - 1] += 1
                        else:
                            self.T[state, action, state1] += 1
                        self.R[state, action] += reward
                    self.R[state, action] /= np.sum(self.T[state, action, :])
                    self.T[state, action, :] /= np.sum(self.T[state, action, :])
                    # print(self.T[state, action, :])
            np.save(file_path, {'S': self.S, 'A': self.A, 'T': self.T, 'R': self.R, 'gamma': self.gamma})
            print('Finished!')

    def ob_to_state(self, ob):
        x, x_dot, theta, theta_dot = ob
        x_idx = np.where(self.x_slices <= x)[0][-1]
        x_dot_idx = np.where(self.x_dot_slices <= x_dot)[0][-1]
        theta_idx = np.where(self.theta_slices <= theta)[0][-1]
        theta_dot_idx = np.where(self.theta_dot_slices <= theta_dot)[0][-1]

        state = x_idx \
                + self.N_X * x_dot_idx \
                + self.N_X * self.N_X_DOT * theta_idx \
                + self.N_X * self.N_X_DOT * self.N_THETA * theta_dot_idx

        return state

    def sample_ob_from_state(self, state):
        x_idx_low = state % self.N_X
        x_idx_high = np.min([x_idx_low + 1, self.N_X - 1])
        x_dot_idx_low = state // self.N_X % self.N_X_DOT
        x_dot_idx_high = np.min([x_dot_idx_low + 1, self.N_X_DOT - 1])
        theta_idx_low = state // self.N_X // self.N_X_DOT % self.N_THETA
        theta_idx_high = np.min([theta_idx_low + 1, self.N_THETA - 1])
        theta_dot_idx_low = state // self.N_X // self.N_X_DOT // self.N_THETA % self.N_THETA_DOT
        theta_dot_idx_high = np.min([theta_dot_idx_low + 1, self.N_THETA_DOT - 1])

        x = np.random.uniform(self.x_slices[x_idx_low], self.x_slices[x_idx_high])
        x_dot = np.random.uniform(self.x_dot_slices[x_dot_idx_low], self.x_dot_slices[x_dot_idx_high])
        theta = np.random.uniform(self.theta_slices[theta_idx_low], self.theta_slices[theta_idx_high])
        theta_dot = np.random.uniform(self.theta_dot_slices[theta_dot_idx_low], self.theta_dot_slices[theta_dot_idx_high])
        ob = (x, x_dot, theta, theta_dot)

        return ob

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = np.clip(x + self.tau * x_dot, -self.high[0], self.high[0])
        x_dot = np.clip(x_dot + self.tau * xacc, -self.high[1], self.high[1])
        theta = np.clip(theta + self.tau * theta_dot, -self.high[2], self.high[2])
        theta_dot = np.clip(theta_dot + self.tau * thetaacc, -self.high[3], self.high[3])
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        info = {'state': np.array(self.state)}
        return self.ob_to_state(self.state), reward, done, info

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return self.ob_to_state(self.state)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def draw_policy_evaluation(self, Q, pi=None):
        pass
