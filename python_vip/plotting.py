import matplotlib.pyplot as plt
import numpy as np

figsize = (20, 10)

def plot_trial_normal(times: np.array, positions: np.array, velocities: np.array, actions: np.array, output_path: str):
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(figsize=(20,10))
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()
    twin2 = ax.twinx()

    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    twin2.spines.right.set_position(("outward", 120))

    p1, = ax.plot(times, actions, "g-", label="Joystick deflections", zorder=100)
    p2, = twin1.plot(times,velocities, "r--", label="theta_dot(t)/angular velocity", zorder=200)
    p3, = twin2.plot(times, positions, "b-", label="theta(t)/angular position", zorder=300)


    positions_max = 65
    actions_max = 1.1
    velocities_max = np.max(np.absolute(velocities)) + 5
    # ax.set_xlim(0, 2)
    ax.set_ylim(-actions_max, actions_max)
    twin1.set_ylim(-velocities_max, velocities_max)
    twin2.set_ylim(-positions_max, positions_max)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joystick Deflection")
    twin1.set_ylabel("Velocity (degrees/sec)")
    twin2.set_ylabel("Position (degrees)")

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)

    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)


    plt.savefig(output_path, dpi=300)



def plot_trial_who_made_action(times: np.array, who_made_the_action: np.array, output_path: str):
    plt.scatter(times, who_made_the_action, )
    plt.xlabel('Time (s)')
    plt.ylabel('Entity')
    plt.title("Plot of which entity made the action over time. 1-pilot, 3-assistant")
    plt.savefig(output_path, dpi=300)


def plot_trial_normal_entities(times: np.array, positions: np.array, pilot_actions: np.array, assistant_actions: np.array, output_path: str):
    fig, ax = plt.subplots(figsize=(20,10))
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()
    twin2 = ax.twinx()

    twin2.spines.right.set_position(("outward", 120))

    p1, = ax.plot(times, pilot_actions, "rx-", label="Pilot Joystick Deflections", zorder=100)
    p2, = twin1.plot(times, assistant_actions, "g--", label="Assistant Joystick Deflections", zorder=200)
    p3, = twin2.plot(times, positions, "b-", label="theta(t)/angular position", zorder=300)

    positions_max = 65
    pilot_actions_max = 1.1
    assistant_actions_max = 1.1
    actions_max = max(pilot_actions_max, assistant_actions_max)
    # ax.set_xlim(0, 2)
    ax.set_ylim(-actions_max, actions_max)
    twin1.set_ylim(-actions_max, actions_max)
    twin2.set_ylim(-positions_max, positions_max)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pilot Joystick Deflections")
    twin1.set_ylabel("Assistant Joystick Deflections")
    twin2.set_ylabel("Position (degrees)")

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)

    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)


    plt.savefig(output_path, dpi=300)


def plot_trial_crash_prob(times: np.array, positions: np.array, velocities: np.array, crash_probs: np.array, output_path: str):
    
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(figsize=(20,10))
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()
    twin2 = ax.twinx()

    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    twin2.spines.right.set_position(("outward", 120))

    p1, = ax.plot(times, crash_probs, "g-", label="Crash Probability", zorder=100)
    p2, = twin1.plot(times,velocities, "r--", label="theta_dot(t)/angular velocity", zorder=200)
    p3, = twin2.plot(times, positions, "b-", label="theta(t)/angular position", zorder=300)


    positions_max = 65
    velocities_max = np.max(np.absolute(velocities)) + 5
    ax.set_ylim(0, +1)
    twin1.set_ylim(-velocities_max, velocities_max)
    twin2.set_ylim(-positions_max, positions_max)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Crash Probability")
    twin1.set_ylabel("Velocity (degrees/sec)")
    twin2.set_ylabel("Position (degrees)")

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)

    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)


    plt.savefig(output_path, dpi=300)
