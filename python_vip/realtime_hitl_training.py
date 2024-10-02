from multiprocessing import freeze_support
import sys

sys.path.append("../")
import os
import subprocess
from pathlib import Path


import pandas as pd
import numpy as np
import json
import glob


import argparse

import matplotlib.pyplot as plt

from hitl_retraining_utils import * 


def plot_performance(study_name, ppt_id, run_mode, model_id):

    if run_mode != "ai_training":
        pth_to_data = f"../output/{study_name}/{ppt_id}/{run_mode}/*/*.csv" 
        pths = sorted(glob.glob(pth_to_data), key=os.path.getmtime)
        first_run = pths[-2]
        last_run = pths[-1]

        first_run_data = pd.read_csv(first_run)
        last_run_data = pd.read_csv(last_run)
    else:
        pth_to_data = f"../output/{study_name}/{ppt_id}/{run_mode}_{model_id}_alone/*/*.csv" 
        pths = sorted(glob.glob(pth_to_data), key=os.path.getmtime)

        first_run = pths[-1]
        pth_to_data = f"../output/{study_name}/{ppt_id}/{run_mode}_{model_id}_retrained_alone/*/*.csv" 
        pths = sorted(glob.glob(pth_to_data), key=os.path.getmtime)
        last_run = pths[-1]

        first_run_data = pd.read_csv(first_run)
        last_run_data = pd.read_csv(last_run)
    check_anticipatory_deflection = lambda x: 1 if x[0]!=0 and np.sign(x[1])!=np.sign(x[2])  else 0
    assign_label = lambda x,y: 0 if (x == y) or (not x and not y) else 1 if x == 1 else 2

    first_run_data['anticipatory_deflections'] = first_run_data.apply(lambda x: check_anticipatory_deflection([x[1], x[2], x[3]]), axis=1)
    first_run_data['deflection_type_label'] = first_run_data.apply(lambda x: assign_label(x[7], x[10]), axis=1)

    last_run_data['anticipatory_deflections'] = last_run_data.apply(lambda x: check_anticipatory_deflection([x[1], x[2], x[3]]), axis=1)
    last_run_data['deflection_type_label'] = last_run_data.apply(lambda x: assign_label(x[7], x[10]), axis=1)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))


    twin1 = axes[0,0].twinx()
    twin2 = axes[0,0].twinx()

    twin2.spines.right.set_position(("outward", 60))

    p1, = axes[0,0].plot(first_run_data["time"], first_run_data["joystick deflection"], "g-", label="Joystick deflections", zorder=100)
    p2, = twin1.plot(first_run_data["time"], first_run_data["angular velocity"], "r--", label="theta_dot(t)/angular velocity", zorder=200)
    p3, = twin2.plot(first_run_data["time"], first_run_data["angular position"], "b-", label="theta(t)/angular position", zorder=300)


    positions_max = 65
    actions_max = 1.1
    velocities_max = np.max(np.absolute(first_run_data["angular velocity"])) + 5
    # ax.set_xlim(0, 2)
    axes[0,0].set_ylim(-actions_max, actions_max)
    twin1.set_ylim(-velocities_max, velocities_max)
    twin2.set_ylim(-positions_max, positions_max)

    axes[0,0].set_xlabel("Time (s)")
    axes[0,0].set_ylabel("Joystick Deflection")
    twin1.set_ylabel("Velocity (degrees/sec)")
    twin2.set_ylabel("Position (degrees)")

    axes[0,0].yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.1)

    axes[0,0].tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    axes[0,0].tick_params(axis='x', **tkw)


    for i, x in enumerate([("other", "black"), ("destab", "red"), ("anticip", "blue"),]):
        label, color = x
        axes[0,1].scatter(first_run_data[first_run_data["deflection_type_label"] == i]["angular position"], first_run_data[first_run_data["deflection_type_label"] == i]["angular velocity"], marker=".", c=color, label=label, alpha=0.7)

    axes[0,1].set_xlabel("Angular Position (θ)")
    axes[0,1].set_ylabel("Angular Velocity (θ/s)")
    axes[0,1].set_xlim((-62, 62))
    axes[0,1].legend()
    axes[0,1].yaxis.tick_right()
    axes[0,1].yaxis.set_label_position("right")


    # second run data

    twin1 = axes[1,0].twinx()
    twin2 = axes[1,0].twinx()

    twin2.spines.right.set_position(("outward", 60))

    p1, = axes[1,0].plot(last_run_data["time"], last_run_data["joystick deflection"], "g-", label="Joystick deflections", zorder=100)
    p2, = twin1.plot(last_run_data["time"], last_run_data["angular velocity"], "r--", label="theta_dot(t)/angular velocity", zorder=200)
    p3, = twin2.plot(last_run_data["time"], last_run_data["angular position"], "b-", label="theta(t)/angular position", zorder=300)


    positions_max = 65
    actions_max = 1.1
    velocities_max = np.max(np.absolute(last_run_data["angular velocity"])) + 5
    # ax.set_xlim(1, 2)
    axes[1,0].set_ylim(-actions_max, actions_max)
    twin1.set_ylim(-velocities_max, velocities_max)
    twin2.set_ylim(-positions_max, positions_max)

    axes[1,0].set_xlabel("Time (s)")
    axes[1,0].set_ylabel("Joystick Deflection")
    twin1.set_ylabel("Velocity (degrees/sec)")
    twin2.set_ylabel("Position (degrees)")

    axes[1,0].yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.1)

    axes[1,0].tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    axes[1,0].tick_params(axis='x', **tkw)


    for i, x in enumerate([("other", "black"), ("destab", "red"), ("anticip", "blue"),]):
        label, color = x
        axes[1,1].scatter(last_run_data[last_run_data["deflection_type_label"] == i]["angular position"], last_run_data[last_run_data["deflection_type_label"] == i]["angular velocity"], marker=".", c=color, label=label, alpha=0.7)

    axes[1,1].set_xlabel("Angular Position (θ)")
    axes[1,1].set_ylabel("Angular Velocity (θ/s)")
    axes[1,1].set_xlim((-62, 62))
    axes[1,1].legend()
    axes[1,1].yaxis.tick_right()
    axes[1,1].yaxis.set_label_position("right")

    plt.tight_layout()
    plt.show()

    plt.savefig(f"../output/{study_name}/{ppt_id}/{run_mode}_{model_id}_performance.png", dpi=300)




def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        prog="HITL tool Runner",
        description="This script runs the HITL tool, where either a human can be trained with an AI assistant or an AI model can be updated with human feedback",
    )

    parser.add_argument("--mode", required=True, type=str, choices=["pre_demo", "human_training", "ai_training"])
    parser.add_argument("--study_name", required=True, type=str)
    parser.add_argument("--ppt_id", required=True, type=str)
    parser.add_argument("--eval_mode", required=False, action="store_true")
    parser.add_argument("--model_id", required=False, type=str, default="A")

    args = parser.parse_args()

    study_name = args.study_name
    ppt_id = args.ppt_id
    model_id = args.model_id
    eval_mode = args.eval_mode
    run_mode = args.mode


    model_details_path = "./name_mappings.json"

    with open(model_details_path, "r") as f:
        model_details = json.load(f)


    assistant_model_details = model_details["models"][model_id]

    crash_prediction_args = " --crash_model_path ../working_models/crash_prediction/model_1000ms_window_800ms_ahead/model --crash_model_norm_stats ../working_models/crash_prediction/model_1000ms_window_800ms_ahead/normalization_mean_std.pkl --crash_pred_window 1"
    working_dir = "<path_to_dir>/python_vip"


    if run_mode == "pre_demo":
        """
        run human pendulum then rdk

        print performance
        """
        
        command_pre_demo = f"python vip.py --study_name {study_name} --ppt_id {ppt_id} --experiment_name {run_mode} --show_metadata --show_crash_bounds" + crash_prediction_args + f" --protocol ./human_trials_1_protocols/pre_demo.csv" + " --use_joystick" + (" --eval_mode" if eval_mode else "")

        print(f"Running pre-demo mode")

        os.system(f"cd {working_dir} && {command_pre_demo}")

        plot_performance(study_name, ppt_id, run_mode, model_id)

    elif run_mode == "human_training":
        """
        run human with rdk 
        then run with AI suggesting
        """
        
        command_human = f"python vip.py --study_name {study_name} --ppt_id {ppt_id} --experiment_name {run_mode} " + crash_prediction_args + f" --protocol ./human_trials_1_protocols/rdk_hard.csv" + " --use_joystick" + (" --eval_mode" if eval_mode else "")

        Path(f"../configs/simulation/{study_name}/{ppt_id}/").mkdir(
            parents=True, exist_ok=True
        )

        config = {
            "pyvip_config": {
                "pilot": {"name": ppt_id, "type": "human"},
                "assistant": assistant_model_details,
                "crash_predictor": {
                    "model_path": "../working_models/crash_prediction/model_1000ms_window_800ms_ahead/model",
                    "norms_path": "../working_models/crash_prediction/model_1000ms_window_800ms_ahead/normalization_mean_std.pkl",
                    "window_size_in_seconds": 1.0,
                },
                "strategy": "suggestion",
                "SAFE_ZONE": 0.2,
                "DANGER_ZONE": 0.25,
                "NOISE_DEVIATION_SUGGESTION": 0.05,
                "ASSISTANT_CONFIDENCE": 0.8,
                "HUMAN_REACTION_TIME": 0.4,
                "HUMAN_REACTION_TIME_NOISE": 0.05,
                "CRASH_PROB_THRESHOLD": 0.8,
            }
        }

        with open(f'../configs/simulation/{study_name}/{ppt_id}/demo_config.json', 'w') as f:
            json.dump(config, f)


        command_human_with_ai = f"python vip.py --study_name {study_name} --ppt_id {ppt_id} --experiment_name {run_mode} --protocol ./human_trials_1_protocols/rdk_hard.csv --experiment_config ../configs/simulation/{study_name}/{ppt_id}/demo_config.json" + (" --eval_mode" if eval_mode else "")

        print(f"Running human training mode")
        os.system(f"cd {working_dir} && {command_human}")
        os.system(f"cd {working_dir} && {command_human_with_ai}")
        plot_performance(study_name, ppt_id, run_mode, model_id)


    elif run_mode == "ai_training":

        """
        run AI alone
        run with HITL 
        retrain with disagreement data
        run retrained AI alone
        """
        inp = "no"
        model_path = assistant_model_details["path"]

        command_ai_alone = f"python vip.py --study_name {study_name} --ppt_id {ppt_id} --experiment_name {run_mode}_{model_id}_alone --show_metadata --show_crash_bounds" + crash_prediction_args + f" --protocol ./human_trials_1_protocols/rdk_hard.csv " + f" --model_type {assistant_model_details['type']} --model_path {model_path} --model_window_size {assistant_model_details['window_size']}" + (" --eval_mode" if eval_mode else "")

        print(f"Running {assistant_model_details['name']} model alone")
        os.system(f"cd {working_dir} && {command_ai_alone}")

        while inp == "no":
            print("Running HITL trial with AI in control and human suggesting in HITL mode")

            command_hitl = f"python vip.py --study_name {study_name} --ppt_id {ppt_id} --experiment_name {run_mode}_{model_id}_HITL --run_hitl --use_joystick" + crash_prediction_args + f" --protocol ./human_trials_1_protocols/rdk_hard.csv" + f" --model_type {assistant_model_details['type']} --model_path {model_path} --model_window_size {assistant_model_details['window_size']}" + (" --eval_mode" if eval_mode else "")
            
            # print(f"\n{command_hitl}\n")
            os.system(f"cd {working_dir} && {command_hitl}")

            retrained_model = retrain_model(study_name, ppt_id, run_mode, model_id, assistant_model_details)
            if retrained_model is None:
                print("No retraining done as no disagreement data found OR model type not supported for retraining, exiting script...")
                return
            else:
                print(f"Retrained model saved at {retrained_model}")

            command_retrained_ai_alone = f"python vip.py --study_name {study_name} --ppt_id {ppt_id} --experiment_name {run_mode}_{model_id}_retrained_alone --show_metadata --show_crash_bounds" + crash_prediction_args + f" --protocol ./human_trials_1_protocols/rdk_hard.csv" + f" --model_type {assistant_model_details['type']} --model_path {retrained_model} --model_window_size {assistant_model_details['window_size']}" + (" --eval_mode" if eval_mode else "")   

            os.system(f"cd {working_dir} && {command_retrained_ai_alone}")

            plot_performance(study_name, ppt_id, run_mode, model_id)

            inp = input("If model performance was satisfactory, enter yes to exit, else press no to continue training: ")
            while inp not in ["yes", "no"]:
                inp = input("If model performance was satisfactory, enter yes to exit, else press no to continue training: ")
            
            if inp == "yes":
                print("New model saved at ", retrained_model)
                exit()
            else:
                print("Continuing training...")
                model_path = retrained_model
        print(f"\n\nUpdated model saved at {retrained_model}")


if __name__ == "__main__":
    freeze_support()
    main()
