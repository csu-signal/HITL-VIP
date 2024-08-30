from functools import lru_cache, partial
from typing import Iterable, Optional, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset, Features, Sequence, Value
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.dataset.pandas import PandasDataset
from gluonts.itertools import Cached, Cyclic, Map
from gluonts.time_feature import (
    TimeFeature,
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    RenameFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform.sampler import InstanceSampler, PredictionSplitSampler
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PretrainedConfig

from deep_learning.networks.gru import gru_regressor
from deep_learning.networks.lstm import lstm_regressor
from deep_learning.networks.mlp_regressor import mlp_regressor
from deep_learning.networks.rnn import rnn_regressor

from .expert_dataset import ExpertDataSet
from .mars_dataset import MarsDataset

#### Basic Data filtering/cleaning.


def filter_data_based_proficiency(
    data: pd.DataFrame, proficiencies: pd.DataFrame, proficiency: list[str] = ["Bad"]
):
    proficiencies = proficiencies[proficiencies.overall_proficiency.isin(proficiency)]

    ppt_ids = []
    ppts = []
    for i, ppt in enumerate(proficiencies.ppt_id.values):
        ppt_ids.extend(
            [
                "1_{}".format(ppt),
                "2_{}".format(ppt),
                "1_{}/".format(ppt),
                "2_{}/".format(ppt),
                ppt,
            ]
        )
        ppts.append(ppt)

    filtered_data = data[data.peopleName.isin(ppt_ids)]

    return ppts, filtered_data


def train_test_split_trials(data: pd.DataFrame, ppts: list, train: int = 0.9):
    train_trials, test_trials = [], []

    for ppt in ppts:
        ppt_data = data[data.peopleName.str.contains(ppt, case=False)]

        ppt_trials = list(sorted(ppt_data.peopleTrialKey.unique()))

        len_train_trials = int(len(ppt_trials) * train)
        train_trials.extend(ppt_trials[:len_train_trials])
        test_trials.extend(ppt_trials[len_train_trials:])

    train_data = data[data.peopleTrialKey.isin(train_trials)]
    test_data = data[data.peopleTrialKey.isin(test_trials)]

    return train_data, test_data


def get_observations_actions_arrays(
    dataframe: pd.DataFrame,
    proficiencies: pd.DataFrame,
    parameters: dict,
    train: bool = True,
    train_fraction: float = 0.9,
):
    ppts, ppt_data = filter_data_based_proficiency(
        dataframe, proficiencies, proficiency=parameters["proficiency"]
    )

    train_data, test_data = train_test_split_trials(
        ppt_data, ppts, train=train_fraction
    )

    ppt_data = train_data if train else test_data
    ppt_data = ppt_data[
        ppt_data["trialPhase"] == 3
    ]  # filter out reset and crash phases

    seconds = ppt_data[["seconds"]].to_numpy()

    ppt_data = ppt_data.drop(columns=["peopleTrialKey", "peopleName"])
    ppt_data["x_value"] = ppt_data["currentPosRollRadians"].apply(
        lambda theta: np.cos(theta)
    )
    ppt_data["y_value"] = ppt_data["currentPosRollRadians"].apply(
        lambda theta: np.sin(theta)
    )

    obs_df = ppt_data.drop(
        columns=["seconds", "trialPhase", "currentPosRollRadians", "joystickX"]
    )[["x_value", "y_value", "currentVelRollRadians"]]
    act_df = ppt_data[["joystickX"]]

    obs_arr, act_arr = obs_df.to_numpy(), act_df.to_numpy()
    return obs_arr, act_arr, seconds


#### Helpers for transformers data processing

# From here we have to map the pandas dataset's `start` field into a time stamp instead of a `pd.Period`. We do this by defining the following class:


class ProcessStartField:
    ts_id = 0

    def __call__(self, data):
        data["start"] = data["start"].to_timestamp()
        data["feat_static_cat"] = [self.ts_id]
        self.ts_id += 1

        return data


@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)


def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch


def process_timeseries(
    data: pd.DataFrame,
    rolling_window_size: int,
    window_size: int,
    prediction_length: int,
    train: bool = True,
) -> pd.DataFrame:
    # process data into windows of size `window_size` after every `rolling_window_size`
    # if train is false, means we are creating the test dataset which has of sizes `window_size + prediction_length`

    processed_data = []
    effective_window_size = window_size if train else window_size + prediction_length

    trials = data.peopleTrialKey.unique()
    for trial in tqdm(trials):
        trial_data_values = data[data["peopleTrialKey"] == trial].values

        # columns = seconds, trialPhase, currentPosRollRadians, currentVelRollRadians, joystickX, peopleName, peopleTrialKey

        length_dataframe = len(trial_data_values)
        index = 0

        window_num = 0

        while index < length_dataframe:
            is_invalid_window = (
                trial_data_values[index, 1] != 3
                or not (index + effective_window_size < length_dataframe)
                or (
                    4 in trial_data_values[index : index + effective_window_size, 1]
                    or 1 in trial_data_values[index : index + effective_window_size, 1]
                )
            )

            if is_invalid_window:
                index += 1

            else:
                temp_data = trial_data_values[index : index + effective_window_size, :]
                temp_df = pd.DataFrame(
                    temp_data,
                    columns=[
                        "seconds",
                        "trialPhase",
                        "currentPosRollRadians",
                        "currentVelRollRadians",
                        "joystickX",
                        "peopleName",
                        "peopleTrialKey",
                    ],
                )
                if len(temp_df) == effective_window_size:
                    temp_df["peopleTrialKey_window_num"] = temp_df[
                        "peopleTrialKey"
                    ].apply(lambda x: f"{x}_{window_num}")

                    processed_data.append(temp_df)
                    window_num += 1
                    index += rolling_window_size
                else:
                    index += 1

    processed_df = pd.concat(processed_data)

    return processed_df


def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    # a bit like torchvision.transforms.Compose
    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # month of year in the case when freq="M"
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=[],
                pred_length=config.prediction_length,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in the life the value of the time series is
            # sort of running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )


def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )


def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the 366 possible transformed time series)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream, is_train=True)

    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )


def create_test_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # we create a Test Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "test")

    # we apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)

    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )


##### plotting functions


def plot_predicted_joystick_with_actual_trial(
    model: Union[
        mlp_regressor, rnn_regressor, lstm_regressor, gru_regressor, SAC, DDPG
    ],
    dataset: Union[MarsDataset, pd.DataFrame],
    data_processing_parameters: dict,
    device: torch.device = torch.device("cpu"),
    proficiencies: pd.DataFrame = None,
    save: bool = False,
    prof: str = None,
):
    if data_processing_parameters["type"] in ["mlp", "rnn", "lstm", "gru"]:
        Ts = dataset.targets
        features = dataset.features_physics.float()
        if dataset.lstm_format:
            Xs = features.reshape(
                -1, data_processing_parameters["past_window_size"] + 1, 3
            )
        else:
            Xs = features

        model = model.to(device)

        Ys = model.use(Xs, device=device)

        positions = np.degrees(features[:, -3].numpy())
        velocities = np.degrees(features[:, -2].numpy())
        actual_joysticks = Ts.numpy()
        predicted_joysticks = Ys
        times = np.float32(dataset.metas[:, 0])

    else:
        test_observations, actual_joysticks, times = get_observations_actions_arrays(
            dataset,
            proficiencies,
            data_processing_parameters,
            train=False,
            train_fraction=0,
        )

        predicted_joysticks, _ = model.predict(
            torch.tensor(test_observations, device="cpu", dtype=torch.float32)
        )

        velocities = np.degrees(test_observations[:, 2])
        positions = np.degrees(
            np.arctan2(test_observations[:, 1], test_observations[:, 0])
        )
        # times = np.float32(dataset.seconds[:,0])

    print(
        f"mean absolute error: {mean_absolute_error(actual_joysticks, predicted_joysticks)}  with max error of  {max_error(actual_joysticks, predicted_joysticks)}"
    )

    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(30, 10), dpi=300)
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()
    twin2 = ax.twinx()

    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    twin2.spines.right.set_position(("axes", 1.2))

    (p1,) = ax.plot(
        times,
        positions,
        "-",
        color="black",
        label="theta(t)/angular position",
        zorder=30,
    )
    (p2,) = twin1.plot(
        times,
        actual_joysticks,
        "x-",
        color="red",
        label="Actual joystick deflection",
        zorder=15,
    )
    (p3,) = twin2.plot(
        times,
        predicted_joysticks,
        "-.",
        color="green",
        label="Predicted joystick deflections",
        zorder=0,
    )

    positions_max = np.max(np.absolute(positions)) + 0.1
    actual_joysticks_max = np.max(np.absolute(actual_joysticks)) + 0.1
    predicted_joysticks_max = np.max(np.absolute(predicted_joysticks)) + 0.1
    # ax.set_xlim(0, 2)
    ax.set_ylim(-positions_max, positions_max)
    twin1.set_ylim(-predicted_joysticks_max, predicted_joysticks_max)
    twin2.set_ylim(-actual_joysticks_max, actual_joysticks_max)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (radians)")
    twin1.set_ylabel("Actual Joystick deflections")
    twin2.set_ylabel("Predicted Joystick deflections")

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)

    ax.tick_params(axis="y", colors=p1.get_color(), **tkw)
    twin1.tick_params(axis="y", colors=p2.get_color(), **tkw)
    twin2.tick_params(axis="y", colors=p3.get_color(), **tkw)
    ax.tick_params(axis="x", **tkw)

    if save:
        plt.savefig(
            f"./output/{data_processing_parameters['name']}/prof_{prof}_trial.png",
            dpi=300,
        )
