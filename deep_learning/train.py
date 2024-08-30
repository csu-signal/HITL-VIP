import sys

sys.path.append("../")

# This notebook is for training deep learning models using the MARS dataset.

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
    args = sys.argv

    if len(args) < 2:
        print(f"Usage: python train.py <train_config_path> <optional - gpu_num>")
        sys.exit(1)

    elif len(args) == 2:
        args.append(0)

    data_path = "../data/"

    training_parameters = None

    config_path = args[1]
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

    # Data processing for ML models
    # ML models e.g. LSTMs and MLPs use different datastructures for input, here we will define PyTorch datasets that will handle the data processing to be used for training and testing our ML models.

    train_ml_dataset = MarsDataset(
        train_data,
        past_window_size=training_parameters["past_window_size"],
        future_time=training_parameters["future_time"],
        lstm_format=training_parameters["lstm_format"],
    )

    test_ml_dataset = MarsDataset(
        test_data,
        past_window_size=training_parameters["past_window_size"],
        future_time=training_parameters["future_time"],
        lstm_format=training_parameters["lstm_format"],
        Xmeans=train_ml_dataset.Xmeans,
        Xstds=train_ml_dataset.Xstds,
        Tmeans=train_ml_dataset.Tmeans,
        Tstds=train_ml_dataset.Tstds,
    )

    train_ml_loader = DataLoader(
        train_ml_dataset,
        batch_size=training_parameters["hyperparameters"]["batch_size"],
        shuffle=True,
        num_workers=10,
    )
    test_ml_loader = DataLoader(
        test_ml_dataset,
        batch_size=training_parameters["hyperparameters"]["batch_size"],
        shuffle=False,
        num_workers=10,
    )

    # Training the model
    # Using Pytorch lightning we will train the models save the best model for use in the experiment later.

    joystick_model = None

    if training_parameters["type"] == "mlp":
        joystick_model = mlp_regressor(
            training_parameters["architecture"]["n_inputs"]
            * (training_parameters["past_window_size"] + 1),
            training_parameters["architecture"]["hidden_layers"],
            training_parameters["architecture"]["n_outputs"],
            Xmeans=train_ml_dataset.Xmeans,
            Xstds=train_ml_dataset.Xstds,
            Tmeans=train_ml_dataset.Tmeans,
            Tstds=train_ml_dataset.Tstds,
            learning_rate=training_parameters["hyperparameters"]["learning_rate"],
            optimizer=training_parameters["hyperparameters"]["optimizer"],
            act_func=training_parameters["hyperparameters"]["activation_function"],
        )

    elif training_parameters["type"] == "rnn":
        joystick_model = rnn_regressor(
            training_parameters["architecture"]["n_inputs"],
            training_parameters["architecture"]["recurrent_hidden_size"],
            training_parameters["architecture"]["recurrent_num_layers"],
            training_parameters["architecture"]["fc_n_hiddens_per_layer"],
            training_parameters["architecture"]["n_outputs"],
            Xmeans=train_ml_dataset.Xmeans,
            Xstds=train_ml_dataset.Xstds,
            Tmeans=train_ml_dataset.Tmeans,
            Tstds=train_ml_dataset.Tstds,
            learning_rate=training_parameters["hyperparameters"]["learning_rate"],
            optimizer=training_parameters["hyperparameters"]["optimizer"],
            act_func=training_parameters["hyperparameters"]["activation_function"],
            bidirectional=training_parameters["architecture"]["bidirectional"],
            dropout_prob=training_parameters["hyperparameters"]["dropout_prob"],
        )

    elif training_parameters["type"] == "lstm":
        joystick_model = lstm_regressor(
            training_parameters["architecture"]["n_inputs"],
            training_parameters["architecture"]["recurrent_hidden_size"],
            training_parameters["architecture"]["recurrent_num_layers"],
            training_parameters["architecture"]["fc_n_hiddens_per_layer"],
            training_parameters["architecture"]["n_outputs"],
            Xmeans=train_ml_dataset.Xmeans,
            Xstds=train_ml_dataset.Xstds,
            Tmeans=train_ml_dataset.Tmeans,
            Tstds=train_ml_dataset.Tstds,
            learning_rate=training_parameters["hyperparameters"]["learning_rate"],
            optimizer=training_parameters["hyperparameters"]["optimizer"],
            act_func=training_parameters["hyperparameters"]["activation_function"],
            bidirectional=training_parameters["architecture"]["bidirectional"],
            dropout_prob=training_parameters["hyperparameters"]["dropout_prob"],
        )

    elif training_parameters["type"] == "gru":
        joystick_model = gru_regressor(
            training_parameters["architecture"]["n_inputs"],
            training_parameters["architecture"]["recurrent_hidden_size"],
            training_parameters["architecture"]["recurrent_num_layers"],
            training_parameters["architecture"]["fc_n_hiddens_per_layer"],
            training_parameters["architecture"]["n_outputs"],
            Xmeans=train_ml_dataset.Xmeans,
            Xstds=train_ml_dataset.Xstds,
            Tmeans=train_ml_dataset.Tmeans,
            Tstds=train_ml_dataset.Tstds,
            learning_rate=training_parameters["hyperparameters"]["learning_rate"],
            optimizer=training_parameters["hyperparameters"]["optimizer"],
            act_func=training_parameters["hyperparameters"]["activation_function"],
            bidirectional=training_parameters["architecture"]["bidirectional"],
            dropout_prob=training_parameters["hyperparameters"]["dropout_prob"],
        )

    print(f"Model instantiated:\n{joystick_model}")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_MAE",
        dirpath=f"./output/{training_parameters['name']}/",
        filename="best_checkpoint",
        save_top_k=1,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_MAE", min_delta=0.0001, patience=50, mode="min"
    )

    logger = TensorBoardLogger("lightning_logs", name=training_parameters["name"])

    trainer = pl.Trainer(
        max_epochs=training_parameters["hyperparameters"]["epochs"],
        precision=32,
        accelerator="auto",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        devices=[int(args[2])],
    )

    trainer.fit(joystick_model, train_ml_loader, test_ml_loader)

    print(
        f"Training has been completed, run \n`python test.py {config_path}` \nto see how it performs on test trials"
    )
