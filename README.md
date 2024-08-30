# README

## Directory structure

* `configs/`: contains configs used for either training deep learning models or for simulations for models to run with the PyVIP program. 
* `data/`: the data files used for training along with other supplementary data.
* `working_models/`: saved checkpoints/weights of the models used in evaluations.
* `deep_learning/`: code dealing with creating and training deep learning models e.g. mlp, rnn, lstm, gru, informer(transformer).
* `reinforcement/`: code dealing with creating and training reinforcement models e.g. ddpg, sac, behavior cloning (BC), adversarial inverse reinforcement learning (AIRL). 
* `python_vip/`: the PyVIP program, faithful python port of the VIP task. 
* `evaluation/`: the various scripts used for running the PyVIP simulations and metric calculations used. 

For more details, look at the READMEs in the relevant directories.  

## How to run

1. Create a new conda env with python=3.9.16
2. Follow the steps in the `reinforcement/`, `deep_learning/` and `python_vip/` readmes to install the required libraries.
3. Download the original MARS data file from link mentioned in the `data/` directories readme and run the `mars_data_preprocess.ipynb` notebook. 
4. For the reinforcement models, run the notebooks in the following order:
    1. `create_BC_expert_dataset.ipynb`
    2. `create_AIRL_expert_dataset.ipynb`
    3. `train_all_models.ipynb`
    4. `bc_new_model_MARS.ipynb`
    5. `AIRL_MARS.ipynb`
5. Copy the saved models from `reinforcement/saved_models/` to `/working_models/assistants/`
6. For the deep learning models, run the `train.py` for each of the training configs present in `configs/training/`. The informer model is optional to train. 
7. Follow the steps mentioned in `deep_learning/README.md` to copy the saved checkpoints to the relevant directory in `working_models/`.

### Running the digital twins study
8. Create the simulation configs via the commands mentioned in `/configs/README.md` under the `Simulation` section.
9. Get performance statistics for the unassisted pilot models: run `evaluate_all_pilots.sh` following instructions under `evaluation/README.md`. (Optionally, get performance statistics for assistants performing the task solo by running `evaluate_all_assistants.sh` in the same way.
10. Run the different evaluation sets.  These are broken up by proficiency of the pilot: `run_good_evals.sh`, `run_med_evals.sh`, `run_bad_evals.sh`. Order does not matter.
11. Run `batch_file_process.sh` on the outputs of all evaluations. We recommend separating the unassisted pilot evals from assisted pilot evals in different folders for clarity.
12. Run `compare_performance.py` to get the diff metrics between assisted and unassisted pilots.
13. After getting the diff metric files, run `analysis_trends_plot.ipynb` to create plots for that identify trends improvements or deterioration in performance of the pilots.

### Running the human subject study

`cd python_vip` 
`python human_trial_runner.py --study_name human_ai_study --use_crash_predictor --session_num <1 or 2> --ppt_id <id>`

1. `python_vip/name_mappings.json`: contains the details about each model i.e. path, retrained path, input+output details, etc.
2. `python_vip/participant_models_mappings.xlsx`: contains information about which model is given to which participant for session 1 and 2. This will also provide the `id` given to each human subject.

#### Retraining the assistants using HITL (Human in the Loop) data

* follow the notebook `evaluation/retrain_assistants_HITL.ipynb` on retraining the assistants

#### Analyzing the data

* follow the notebook `evaluation/analyze_human_ai_study_data.ipynb` for the analysis part. 
* The output for this part will be present in `evaluation/output/`. 

#### Output for human trials

The output for each human subject would be present in `output/human_ai_study/` and contain a directory for each task. In each task directory will contain 2-4 further directories containing 
1. raw data in the format below:
    * time (seconds), angular position (degrees), angular velocity (degrees/seconds), joystick deflection, action_made_by (pilot or assistant), pilot_actions (without noise), assistant_actions (without noise),destabilizing_actions, crash_probabilities, is_crash_condition_triggered
2. various plots comparing combinations of the above variables through time. 
