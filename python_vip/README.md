# Py-VIP

## Pre-reqs

The following python packages are additionally required along with the packages mentioned in the `deep_learning` and `reinforcement` directories to run the PyVIP program:

- pygame
- scipy


These can be installed using either pip or conda. 
## Usage

`python3 vip.py`

The script requires a `protocols.csv` file which contains some configurations for the VIP trial. 

* fps: the number of frames per second to render the simulation
* trial_time: how long to run the trial in seconds
* position_limit: the limits from the direction of balance from which crash limits are set
* ndots: number of dots to display when RDK version of VIP is running
* coherence: how many dots to keep coherent every iteration
* gravity: g force, default 9.801 m/s^2
* ksp: Torsional spring constant for the pendulum  (N-m/rad)
* length: length of the pendulum in cm 
* bv: Damping of the pendulum, (Nm-s/rad)
* mass: mass of the pendulum in kg
* ipoff: Defines reset range re DOB. Maximum offset of pendulum reset position from DOB, 
specified  as a fraction of fall limit (flim1, see below).  The actual reset  value is randomly
chosen between  ±ipoff*flim1/2  to ipoff, as shown in the figure.

Lines can be added to the protocols file with different values which would depict a new trial with different parameters. If any value is missing in the added lines then those are forward filled. 

The script also has additional arguments that could add a model to control the pendulum

 * --protocol : path to the protocol file, default is './protocols.csv'
 * --model_path : path to the model being used.
 * --model_type : type of the model where options are either sac, ddpg, mlp, rnn, lstm, gru, or informer  
 * --model_window_size : input window size to the model in seconds. Default is 0. 
 * --model_intermittent: percentage of model predicted deflections to zero out or drop. default is 0. max value is 100. 
 * --noise_model_action: amount of random to be added to the deflections predicted by the model. Default is 0. 
 * --eval_mode : passing this argument will not require a spacebar key press to start the experiment or reset it after a crash. without passing this argument you will need to press the spacebar key when prompted.
 * --crash_model_path : path to the crash prediction model
 * --crash_model_norm_stats : path to the input normalization statistics (mean and std) for continuous variables. This is `required if crash_model_path is provided`
 * --crash_pred_window : the input size to the crash prediction model in seconds. default is 1.0 seconds.



### Running the digital twins study simulations

Before running the experiment, make sure the path of models that are being used in the `pyvip_config` section of the experiment config are correct and reachable from the `python_vip` directory. 


The following command will start the pyvip program:

`python vip.py --experiment_config <path_to_config> --eval_mode`

#### Output

By default, the outputs for each run will be saved in an `output/` directory in the root of the Project directory. 
Each output would contain the following:
1. The raw data in csv format containing
    * time (seconds), angular position (degrees), angular velocity (degrees/seconds), joystick deflection, action_made_by (pilot or assistant), pilot_actions (without noise), assistant_actions (without noise),destabilizing_actions, crash_probabilities
2. various plots comparing combinations of the above variables through time. 


### Running the human subject study

`python human_trial_runner.py --study_name human_ai_study --use_crash_predictor --session_num <1 or 2> --ppt_id <id>`

1. `name_mappings.json`: contains the details about each model i.e. path, retrained path, input+output details, etc.
2. `participant_models_mappings.xlsx`: contains information about which model is given to which participant for session 1 and 2. 

#### Output

The output for each participant would be present in "human_ai_study" and contain a directory for each task. In each task directory will contain 2-4 further directories containing 
1. raw data in the format below:
    * time (seconds), angular position (degrees), angular velocity (degrees/seconds), joystick deflection, action_made_by (pilot or assistant), pilot_actions (without noise), assistant_actions (without noise),destabilizing_actions, crash_probabilities, is_crash_condition_triggered
2. various plots comparing combinations of the above variables through time. 
