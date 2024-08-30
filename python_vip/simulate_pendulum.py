import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def myip7(
    t: float,
    th: float,
    ksp: float,
    ma: float,
    Iz: float,
    bv: float,
    gr: float,
    ln: float,
):
    ka = (ma * ln * gr) / (ma * ln**2 + Iz)
    ks = ksp / (ma * ln**2 + Iz)
    kb = bv / (ma * ln**2 + Iz)
    return [th[1], ka * np.sin(th[0]) - ks * th[0] - kb * th[1]]


def simulate_pendulum(
    init_state: list[float],
    sim_time: float,
    ksp: float,
    ma: float,
    Iz: float,
    bv: float,
    gr: float,
    ln: float,
):
    def deriv_func(t, state):
        return myip7(t, state, ksp, ma, Iz, bv, gr, ln)

    sol = solve_ivp(
        deriv_func, [0, sim_time], init_state, t_eval=np.linspace(0, sim_time, 10)
    )
    return sol.t, sol.y


if __name__ == "__main__":
    # Set up initial conditions
    theta0 = 1.0
    theta_dot0 = 0.0
    init_state = [theta0, theta_dot0]

    # Set up simulation parameters
    sim_time = 20.0
    ksp = 2.0
    ma = 70.0
    Iz = 1.0
    bv = 0.1
    gr = 9.81
    ln = 1.0

    # Run simulation
    t, y = simulate_pendulum(init_state, sim_time, ksp, ma, Iz, bv, gr, ln)

    # Plot results
    plt.plot(t, y[0], "b", label="theta(t)")
    plt.plot(t, y[1], "g", label="theta_dot(t)")
    plt.legend(loc="upper right")
    plt.xlabel("Time (s)")
    plt.grid()
    plt.show()
