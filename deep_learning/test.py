import sys

sys.path.append("../")

# This notebook is for testing deep learning models using the MARS dataset.

import json

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils.utils import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python test.py <train_config_path>")
        sys.exit(1)

    data_path = "../data/"
    device = torch.device("cpu")

    config_path = sys.argv[1]
    print(f"config being used is from {config_path}")

    with open(config_path, "r") as f:
        config_json = json.load(f)
    training_parameters = config_json["dl_training_parameters"]

    print(training_parameters)

    # Data Preprocessing
    # Start by filtering data based on proficiency required for the current experiment. The data preprocessing parameters can be changed by updating the json below to set the proficiency required and the train fraction. For these experiments, the trials in the test set will be chosen from last n trials per participant.

    balancing_data_path = f"{data_path}{'data_all_radians' if 'mars' == training_parameters['dataset'] else 'vip_data_all'}.csv"
    proficiencies_data_path = f"{data_path}{'overall_proficiency' if 'mars' == training_parameters['dataset'] else 'vip_overall_proficiency'}.csv"

    print("Data file used: ", balancing_data_path)
    print("Proficiency file used: ", proficiencies_data_path)

    data = pd.read_csv(balancing_data_path)

    if "mars" == training_parameters["dataset"]:
        data["joystickX"] = -1 * data["joystickX"]

    proficiencies = pd.read_csv(proficiencies_data_path)

    ppts, ppt_data = filter_data_based_proficiency(
        data, proficiencies, proficiency=training_parameters["proficiency"]
    )
    train_data, test_data = train_test_split_trials(
        ppt_data, ppts, train=training_parameters["train_fraction"]
    )

    checkpoint_path = f"./output/{training_parameters['name']}/best_checkpoint.ckpt"
    print(f"Loading checkpoint from {checkpoint_path}")

    joystick_model = None
    if training_parameters["type"] == "mlp":
        joystick_model = mlp_regressor.load_from_checkpoint(
            checkpoint_path, map_location=device
        )

    elif training_parameters["type"] == "rnn":
        joystick_model = rnn_regressor.load_from_checkpoint(
            checkpoint_path, map_location=device
        )

    elif training_parameters["type"] == "lstm":
        joystick_model = lstm_regressor.load_from_checkpoint(
            checkpoint_path, map_location=device
        )

    elif training_parameters["type"] == "gru":
        joystick_model = gru_regressor.load_from_checkpoint(
            checkpoint_path, map_location=device
        )

    print(f"Model instantiated:\n{joystick_model}")

    joystick_model, joystick_model.Xmeans.shape, joystick_model.device

    test_trials = {
        "mars": {
            "Bad": "2_xz_P11/20_600back_Block5_trial_020.csv",
            "Medium": "2_xm_P20/20_600back_Block5_trial_020.csv",
            "Good": "2_rv_P7/20_600back_Block5_trial_020.csv",
        },
        "vip": {"Bad": "S7_T8", "Medium": "S3_T8", "Good": "S26_T8"},
    }

    for prof in training_parameters["proficiency"]:
        test_ppt_trial = test_trials[training_parameters["dataset"]][prof]

        if training_parameters["lstm_format"]:
            test_ppt_dataset = MarsDataset(
                test_data[test_data.peopleTrialKey == test_ppt_trial],
                past_window_size=training_parameters["past_window_size"],
                future_time=training_parameters["future_time"],
                lstm_format=training_parameters["lstm_format"],
                Xmeans=joystick_model.Xmeans.reshape(1, -1),
                Xstds=joystick_model.Xstds.reshape(1, -1),
                Tmeans=joystick_model.Tmeans.reshape(1, -1),
                Tstds=joystick_model.Tstds.reshape(1, -1),
            )
        else:
            test_ppt_dataset = MarsDataset(
                test_data[test_data.peopleTrialKey == test_ppt_trial],
                past_window_size=training_parameters["past_window_size"],
                future_time=training_parameters["future_time"],
                lstm_format=training_parameters["lstm_format"],
                Xmeans=joystick_model.Xmeans,
                Xstds=joystick_model.Xstds,
                Tmeans=joystick_model.Tmeans,
                Tstds=joystick_model.Tstds,
            )

        plot_predicted_joystick_with_actual_trial(
            joystick_model,
            test_ppt_dataset,
            training_parameters,
            device=device,
            save=True,
            prof=prof,
        )
