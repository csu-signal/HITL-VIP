import math

import numpy as np
from simulate_pendulum import simulate_pendulum


class Pendulum:
    def __init__(
        self,
        length: float,
        mass: float,
        times: list[int],
        positions: list[float],
        velocities: list[float],
        joystick_actions: list[float],
        destab_actions: list[float],
        saved_states: list[list[float]],
        theta: float = None,
        time_elapsed: int = 0,
        ipoff: float = 0.25,
    ):
        self.length = length
        self.mass = mass
        self.times = times
        self.positions = positions
        self.velocities = velocities
        self.joystick_actions = joystick_actions
        self.destab_actions = destab_actions
        self.saved_states = saved_states

        if theta is None:
            # self.theta = np.random.uniform(-math.pi/4, math.pi/4)
            self.theta = (
                ((np.random.rand() + 1) * np.sign(np.random.rand() - 0.5)) * ipoff * 0.5
            )
            self.theta_dot = -np.sign(self.theta) * (0.25 + 0.25 * np.random.rand())
        else:
            self.theta = theta
        self.theta_dot = 0

        self.times.append(time_elapsed)
        self.positions.append(self.theta)
        self.velocities.append(self.theta_dot)
        self.joystick_actions.append(0)
        self.destab_actions.append(0)
        self.saved_states.append([self.theta_dot, self.theta, 0, 0])

    def update(
        self,
        base_x: float,
        base_y: float,
        dt: float,
        time_elapsed: int,
        ksp: float,
        bv: float,
        gr: float,
        base_height: float,
    ):
        ln = self.length / 100
        ma = self.mass
        Iz = self.mass * ln**2 / 3

        init_state = [self.theta, self.theta_dot]
        sim_time = dt

        t, y = simulate_pendulum(init_state, sim_time, ksp, ma, Iz, bv, gr, ln)

        self.theta = y[0][-1]
        self.theta_dot = y[1][-1]

        self.times.append(time_elapsed)
        self.positions.append(self.theta)
        self.velocities.append(self.theta_dot)

        pendulum_tip_x = base_x + ln * 100 * math.sin(self.theta)
        pendulum_tip_y = base_y - base_height / 2 - (ln * 100) * math.cos(self.theta)

        return pendulum_tip_x, pendulum_tip_y
