import sys

sys.path.append("../../")

import time
from os import path

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from python_vip.simulate_pendulum import simulate_pendulum

DEFAULT_X = np.pi / 3  # range for starting theta
DEFAULT_Y = np.radians(300)  # range for starting thetadot


class PendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=9.81):
        self.max_speed = np.radians(300)
        self.max_deflect = 1.0
        self.dt = 0.02  # to match MARS data
        self.g = g
        self.m = 70.0
        self.l = 1.0
        self.viewer = None

        self.timer = time.perf_counter()
        self.timer_done = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.max_deflect]),
            high=np.array([self.max_deflect]),
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        ksp = 0
        Iz = self.m * self.l**2 / 3
        bv = 0

        u = np.clip(u, -self.max_deflect, self.max_deflect)[0]
        self.last_u = u  # for rendering
        if th <= (np.pi / 3) / 2 and th >= (-(np.pi / 3)) / 2:
            costs = 0
        else:
            costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        t, y = simulate_pendulum(self.state, dt, ksp, m, Iz, bv, g, l)

        newth = y[0][-1]
        newthdot = y[1][-1] + (u * np.radians(19))
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        terminated = not bool(newth <= np.pi / 3 and newth >= -(np.pi / 3))
        cur_time = time.perf_counter()
        elapsed_time = cur_time - self.timer
        if elapsed_time >= 45.0:
            terminated = True
            self.timer_done = True

        return self._get_obs(), -costs, terminated, {}

    def reset(self):
        high = np.array([DEFAULT_X, DEFAULT_Y])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None

        if self.timer_done:
            self.timer = time.perf_counter()
            self.timer_done = False

        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u is not None:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
