# USAGE (assumes you're in python_vip folder)
# run python ../evaluation/make_configs.py -f ../evaluation/evals.csv -o ../configs/simulation

import argparse
import numpy as np
import pandas as pd
import json

def main():
    parser = argparse.ArgumentParser(
                    prog='make_configs',
                    description='Generate experimental configuration files from CSV')
    parser.add_argument('-f', '--filename')
    parser.add_argument('-o', '--output_dir')
    args = parser.parse_args()
    print(f"Opening {args.filename}")
    
    df = pd.read_csv(args.filename)
    
    print(df)
        
    configs =\
    [{
        "pyvip_config": {
            "pilot": {
                "name": "",
                "type": "",
                "path": "../working_models/pilots/",
                "model_intermittancy": 0,
                "window_size": 0.0

            },
            "assistant": {
                "name": "",
                "type": "",
                "path": "../working_models/assistants/",
                "model_intermittancy": 0,
                "window_size": 0.0
            },
            "crash_predictor": {
                "model_path": "../working_models/crash_prediction/model_1000ms_window_800ms_ahead/model",
                "norms_path": "../working_models/crash_prediction/model_1000ms_window_800ms_ahead/normalization_mean_std.pkl",
                "window_size_in_seconds": 1.0
            },
            "strategy": "suggestion",

            "SAFE_ZONE": 0.2,
            "DANGER_ZONE": 0.25,
            "NOISE_DEVIATION_SUGGESTION": 0.05,
            "ASSISTANT_CONFIDENCE": 0.8,
            "HUMAN_REACTION_TIME": 0.4,
            "HUMAN_REACTION_TIME_NOISE": 0.05,
            "CRASH_PROB_THRESHOLD": 0.8
        }
    }]*len(df)
    
    for i in range(len(df)):
        number = df["Number"][i]
        pilot_name = df["Pilot"][i].split('.')[0]
        configs[i]["pyvip_config"]["pilot"]["name"] = f"{number:03d}_{pilot_name}"
        
        if "mlp" in pilot_name:
            configs[i]["pyvip_config"]["pilot"]["type"] = "mlp"
        elif "rnn" in pilot_name:
            configs[i]["pyvip_config"]["pilot"]["type"] = "rnn"
        elif "lstm" in pilot_name:
            configs[i]["pyvip_config"]["pilot"]["type"] = "lstm"
        elif "gru" in pilot_name:
            configs[i]["pyvip_config"]["pilot"]["type"] = "gru"
        elif "transformer" in pilot_name:
            configs[i]["pyvip_config"]["pilot"]["type"] = "transformer"
        elif "sac" in pilot_name:
            configs[i]["pyvip_config"]["pilot"]["type"] = "sac"
        elif "ddpg" in pilot_name:
            configs[i]["pyvip_config"]["pilot"]["type"] = "ddpg"
          
        if configs[i]["pyvip_config"]["pilot"]["type"] == "sac" or\
                configs[i]["pyvip_config"]["pilot"]["type"] == "ddpg":
            configs[i]["pyvip_config"]["pilot"]["path"] = f'../working_models/pilots/{pilot_name}.zip'
        else:
            configs[i]["pyvip_config"]["pilot"]["path"] = f'../working_models/pilots/{pilot_name}.ckpt'
         
        if "small_window" in pilot_name:
            if "_future" in pilot_name:
                configs[i]["pyvip_config"]["pilot"]["window_size"] = 0.3
            else:
                configs[i]["pyvip_config"]["pilot"]["window_size"] = 0.2
        else:
            if "mlp" in pilot_name and "_window" not in pilot_name:
                configs[i]["pyvip_config"]["pilot"]["window_size"] = 0.0
            else:
                configs[i]["pyvip_config"]["pilot"]["window_size"] = 0.5
            
        asst_name = df["Assistant"][i].split('.')[0]
        configs[i]["pyvip_config"]["assistant"]["name"] = asst_name
        
        if "mlp" in asst_name:
            configs[i]["pyvip_config"]["assistant"]["type"] = "mlp"
        elif "rnn" in asst_name:
            configs[i]["pyvip_config"]["assistant"]["type"] = "rnn"
        elif "lstm" in asst_name:
            configs[i]["pyvip_config"]["assistant"]["type"] = "lstm"
        elif "gru" in asst_name:
            configs[i]["pyvip_config"]["assistant"]["type"] = "gru"
        elif "transformer" in asst_name:
            configs[i]["pyvip_config"]["assistant"]["type"] = "transformer"
        elif "sac" in asst_name:
            configs[i]["pyvip_config"]["assistant"]["type"] = "sac"
        elif "ddpg" in asst_name:
            configs[i]["pyvip_config"]["assistant"]["type"] = "ddpg"
          
        if configs[i]["pyvip_config"]["assistant"]["type"] == "sac" or\
                configs[i]["pyvip_config"]["assistant"]["type"] == "ddpg":
            configs[i]["pyvip_config"]["assistant"]["path"] = f'../working_models/assistants/{asst_name}.zip'
        else:
            configs[i]["pyvip_config"]["assistant"]["path"] = f'../working_models/assistants/{asst_name}.ckpt'
            
        if "small_window" in asst_name:
            if "_future" in asst_name:
                configs[i]["pyvip_config"]["assistant"]["window_size"] = 0.3
            else:
                configs[i]["pyvip_config"]["assistant"]["window_size"] = 0.2
        else:
            if "mlp" in asst_name and "_window" not in asst_name:
                configs[i]["pyvip_config"]["assistant"]["window_size"] = 0.0
            else:
                configs[i]["pyvip_config"]["assistant"]["window_size"] = 0.5
    
        config_file = df["Config file"][i]
        path = f"{args.output_dir}/{config_file}"
        print(f"Writing to {path}")
        f = open(path, "w")
        json.dump(configs[i], f, indent=2)

if __name__ == "__main__":
    main()
