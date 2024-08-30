import sys

sys.path.append("../")
import os
import subprocess
from pathlib import Path


import pandas as pd
import numpy as np
import json


import argparse


def get_model_id(mappings, ppt, session_num):
    if ppt in set(mappings["participant"]):
        # Find the row index where X is present in column A
        row_index = mappings.index[mappings["participant"] == ppt].tolist()[0]
        # Get the corresponding value from column B
        corresponding_value = mappings.loc[row_index, "model"] if session_num == 1 else mappings.loc[row_index, "model_second_session"]
        return corresponding_value
    else:
        print(f"No occurrence of '{ppt}' found in participants columns.")
        return None


def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        prog="Human Trials Runner",
        description="This script runs the vip script given a participant and the assigned model assistant",
        epilog="IDK what this does",
    )

    parser.add_argument("--study_name", required=True, type=str)
    parser.add_argument("--ppt_id", required=True, type=str)
    parser.add_argument("--session_num", required=True, type=int)
    parser.add_argument("--eval_mode", required=False, action="store_true")
    parser.add_argument("--use_crash_predictor", required=False, action="store_true")
    # parser.add_argument('--model_path', default=None, required=False, type=str)

    args = parser.parse_args()

    study_name = args.study_name
    ppt_id = args.ppt_id
    session_num = args.session_num

    model_details_path = "./name_mappings.json"
    ppt_model_mappings_path = "./participants_models_mappings.xlsx"

    ppt_model_mappings = pd.read_excel(ppt_model_mappings_path)
    with open(model_details_path, "r") as f:
        model_details = json.load(f)

    model_id = get_model_id(ppt_model_mappings, ppt_id, session_num)
    assistant_model_details = model_details["models"][model_id]

    print(model_id)
    print(assistant_model_details)

    use_crash_predictor = args.use_crash_predictor
    crash_prediction_args = " --crash_model_path ../working_models/crash_prediction/model_1000ms_window_800ms_ahead/model --crash_model_norm_stats ../working_models/crash_prediction/model_1000ms_window_800ms_ahead/normalization_mean_std.pkl --crash_pred_window 1"
    eval_mode = args.eval_mode

    """
        if session is 1:
        run 4 different protocols (Pendulum x2, RDK alone x3, RDK AI assistance x4, HITL x3)

        if session is 2
        run 3 different protocols (RDK alone x3, RDK AI assistance before x4, RDK AI assistance after x4)
    """
    commands = []
    working_dir = "path/to/root/directory/python_vip"

    if session_num == 1:
        commands.append(
            f"python vip.py --study_name {study_name} --ppt_id {ppt_id} --experiment_name pendulum --protocol ./human_trials_1_protocols/pendulum.csv"
            + (crash_prediction_args if use_crash_predictor else "")
            + (" --eval_mode" if eval_mode else "") 
            + " --monitor 1"
        )

        commands.append(
            f"python vip.py --study_name {study_name} --ppt_id {ppt_id} --experiment_name rdk_alone_first --protocol ./human_trials_1_protocols/rdk_alone.csv"
            + (crash_prediction_args if use_crash_predictor else "")
            + (" --eval_mode" if eval_mode else "")
            + " --monitor 1"
        )

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

        with open(f'../configs/simulation/{study_name}/{ppt_id}/rdk_alone_ai_assist_first.json', 'w') as f:
            json.dump(config, f)

        commands.append(
            f"python vip.py --study_name {study_name} --ppt_id {ppt_id} --experiment_name rdk_alone_AI_assist_first --protocol ./human_trials_1_protocols/rdk_alone_ai_assist.csv --experiment_config ../configs/simulation/{study_name}/{ppt_id}/rdk_alone_ai_assist_first.json"
            + (" --eval_mode" if eval_mode else "")
            + " --monitor 1"
        )

        commands.append(
            f"python vip.py --study_name {study_name} --ppt_id {ppt_id} --experiment_name rdk_alone_hitl --protocol ./human_trials_1_protocols/rdk_alone_hitl.csv --run_hitl --model_path {assistant_model_details['path']} --model_type {assistant_model_details['type']} --model_window_size {assistant_model_details['window_size']}" 
            + (crash_prediction_args if use_crash_predictor else "")
            + (" --eval_mode" if eval_mode else "")
            + " --monitor 1"
        )

    elif session_num == 2:
        commands.append(
            f"python vip.py --study_name {study_name} --ppt_id {ppt_id} --experiment_name rdk_alone_second --protocol ./human_trials_1_protocols/rdk_alone.csv"
            + (crash_prediction_args if use_crash_predictor else "")
            + (" --eval_mode" if eval_mode else "")
            + " --monitor 1"
        )

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

        with open(f'../configs/simulation/{study_name}/{ppt_id}/rdk_alone_ai_assist_second_original.json', 'w') as f:
            json.dump(config, f)

        config["pyvip_config"]["assistant"]["path"] = config["pyvip_config"]["assistant"]["retrained_path"]

        with open(f'../configs/simulation/{study_name}/{ppt_id}/rdk_alone_ai_assist_second_retrained.json', 'w') as f:
            json.dump(config, f)

        commands.append(
            f"python vip.py --study_name {study_name} --ppt_id {ppt_id} --experiment_name rdk_alone_AI_assist_second_original --protocol ./human_trials_1_protocols/rdk_alone_ai_assist.csv --experiment_config ../configs/simulation/{study_name}/{ppt_id}/rdk_alone_ai_assist_second_original.json "
            + (" --eval_mode" if eval_mode else "")
            + " --monitor 1"
        )

        commands.append(
            f"python vip.py --study_name {study_name} --ppt_id {ppt_id} --experiment_name rdk_alone_AI_assist_second_retrained --protocol ./human_trials_1_protocols/rdk_alone_ai_assist.csv --experiment_config ../configs/simulation/{study_name}/{ppt_id}/rdk_alone_ai_assist_second_retrained.json "
            + (" --eval_mode" if eval_mode else "")
            + " --monitor 1"
        )
    else:
        return None
    for command in commands:
        print(command)
        os.system(f"cd {working_dir} && {command}")
        input("Press enter to continue to the next trial")


if __name__ == "__main__":
    main()
