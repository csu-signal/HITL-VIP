# README

## Requirements

python version = 3.9.16

to install gymansium based environment:
```
to install gymnasium: 
    pip install gymnasium
    pip install gymnasium[classic-control]

to install stable-baselines3:
    (per https://stable-baselines3.readthedocs.io/en/master/guide/install.html)
    pip install "stable_baselines3[extra]>=2.0.0a9" 
    (add "--user" flag if a read-only error occurs)
```

```
to install gym based environment:
    to install gym:
        pip install gym
    to install sb3:
        pip install stable-baselines3
        pip install 'shimmy>=0.2.1'
    to install imitation:
        pip install imitation

        replace existing path/to/conda/envs/env_name/lib/python3.9/site-packages/imitation/data/wrappers.py with the included wrappers.py
        replace existing path/to/conda/envs/env_name/lib/python3.9/site-packages/stable_baselines3/common/monitor.py with the included monitor.py
```

## Details

* `envs/` contain all the customized pendulum environments that we used. 
    * `pendulum.py` is the basic pendulum env with only the physics-integration loop updated using logic from the VIP program.
    * `pendulum_fullRandom.py` adds a random starting position within `X degrees` of the direction of balance (DOB) to the above env.
    * `pendulum_piecewiseReward.py` customizes the reward function to provide a reward when it is within a certain `Y degrees` of the DOB
    * `pendulum_penalizeDeflection.py` further adds to the reward function such that it penalizes large actions to reinforce smaller intermittent actions. 
    * `pendulumV21.py` is the `gym` version of the `pendulum_penalizeDeflection.py` env to train the SAC AIRL model.

* `create_BC_expert_dataset.ipynb` creates the Expert Dataset for the Behavior Cloning variants of the SAC and DDPG models.
    * A file called `MARS_BC_expert_transitions.npz` will be created when done. 
* `create_AIRL_expert_transitions.ipynb` creates the Expert Dataset for the SAC AIRL model.
    * A file called `MARS_AIRL_expert_transitions.npz` will be created when done. 

* `train_all_models.ipynb` trains a version of both SAC and DDPG using each of the gymnasium based environments.
* `bc_new_model_MARS.ipynb` trains a version of both SAC and DDPG with the penalizeDeflection env using behavior cloning.
    * update the config json in the file to train the desired model. 
* `AIRL_MARS.ipynb` trains the AIRL algorithm using a SAC model.


All models are saved in a `saved_models/` folder in this directory. 

<!-- MARS_expert_dataset.npz
    compressed npz containing two arrays:
        expert_actions: (756945, 1), MARS joystickX multiplied by -1 
        expert_observations: (756945, 3) [x, y, angular velocity(in rads)] derived from MARS currentPosRoll	and currentVelRoll
    load using:
        data = np.load('MARS_expert_data.npz')
        expert_actions = data['expert_actions']
        expert_observations = data['expert_observations'] -->
