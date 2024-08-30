import sys

sys.path.append("../")

import json

import matplotlib.pyplot as plt
import pandas as pd
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm import tqdm
from transformers import InformerConfig, InformerForPrediction
from utils.utils import *

data_path = "../data/"
config_path = "../configs/training/pilot_mars_informer_good_small_window_future.json"

with open(config_path, "r") as f:
    config_json = json.load(f)
    training_parameters = config_json["dl_training_parameters"]


balancing_data_path = f"{data_path}{'data_all_radians' if 'mars' == training_parameters['dataset'] else 'vip_data_all'}.csv"
proficiencies_data_path = f"{data_path}{'overall_proficiency' if 'mars' == training_parameters['dataset'] else 'vip_overall_proficiency'}.csv"

print("Data file used: ", balancing_data_path)
print("Proficiency file used: ", proficiencies_data_path)


data = pd.read_csv(balancing_data_path)

if "mars" == training_parameters["dataset"]:
    data["joystickX"] = -1 * data["joystickX"]

proficiencies = pd.read_csv(proficiencies_data_path)


epochs = 500


rolling_window_size = training_parameters["rolling_window_size"]
window_size = training_parameters["past_window_size"]
prediction_length = training_parameters["future_time"]
freq = "S"


ppts, ppt_data = filter_data_based_proficiency(
    data, proficiencies, proficiency=training_parameters["proficiency"]
)


print("processing timeseries data")

train_df = process_timeseries(
    ppt_data, rolling_window_size, window_size, prediction_length, train=True
)
test_df = process_timeseries(
    ppt_data, rolling_window_size, window_size, prediction_length, train=False
)

print("length of processed train and test dataframes")
print(
    len(train_df.peopleTrialKey_window_num.unique()),
    len(test_df.peopleTrialKey_window_num.unique()),
)

train_df = train_df[
    train_df.peopleTrialKey_window_num.isin(test_df.peopleTrialKey_window_num)
]

# After converting it into a `pd.Dataframe` we can then convert it into GluonTS's `PandasDataset`:

print("length of processed train and test dataframes after filtering")
print(
    len(train_df.peopleTrialKey_window_num.unique()),
    len(test_df.peopleTrialKey_window_num.unique()),
)


print("creating pandas dataset for further processing")

train_ds = PandasDataset.from_long_dataframe(
    train_df,
    target="joystickX",
    item_id="peopleTrialKey_window_num",
    timestamp="seconds",
    feat_dynamic_real=["currentPosRollRadians", "currentVelRollRadians"],
    freq="ms",
    unchecked=True,
)

test_ds = PandasDataset.from_long_dataframe(
    test_df,
    target="joystickX",
    item_id="peopleTrialKey_window_num",
    timestamp="seconds",
    feat_dynamic_real=["currentPosRollRadians", "currentVelRollRadians"],
    freq="ms",
    unchecked=True,
)

print("train dataset", train_ds)
print("test dataset", test_ds)


print("Processing start field for time series")

process_start = ProcessStartField()

list_train_ds = list(Map(process_start, train_ds))
list_test_ds = list(Map(process_start, test_ds))

# Next we need to define our schema features and create our dataset from this list via the `from_list` function:


features = Features(
    {
        "start": Value("timestamp[s]"),
        "target": Sequence(Value("float32")),
        "feat_static_cat": Sequence(Value("uint64")),
        # "feat_static_real":  Sequence(Value("float64")),
        "feat_dynamic_real": Sequence(Sequence(Value("float32"))),
        # "past_feat_dynamic_real": Sequence(Sequence(Value("float32"))),
        "item_id": Value("string"),
    }
)

print("Creating datasets with out desired features")

train_dataset = Dataset.from_list(list_train_ds, features=features)
test_dataset = Dataset.from_list(list_test_ds, features=features)

train_example = train_dataset[0]
test_example = test_dataset[0]

print("lengths of target in a train and test sample respectively")
print(len(train_example["target"]), len(test_example["target"]))

# We can thus use this strategy to [share](https://huggingface.co/docs/datasets/share) the dataset to the hub.

# ## Training Informer from scratch.


train_dataset.set_transform(partial(transform_start_field, freq=freq))
test_dataset.set_transform(partial(transform_start_field, freq=freq))


lags_sequence = get_lags_for_frequency(freq)
lags_sequence = get_lags_for_frequency(freq, lag_ub=30)
print("Lags sequence to be used ", lags_sequence)

time_features = []
print("Time features to be used", time_features)


config = InformerConfig(
    prediction_length=prediction_length,
    # # context length:
    context_length=prediction_length * 2,
    # input_size = 3,
    # lags coming from helper given the freq:
    lags_sequence=lags_sequence,
    # we'll add 1 time features ("age", see further):
    num_time_features=len(time_features) + 1,
    # we have a single static categorical feature, namely time series ID:
    num_static_categorical_features=1,
    # it has 366 possible values:
    cardinality=[len(train_dataset) * 1000],
    # the model will learn an embedding of size 2 for each of the 366 possible values:
    embedding_dimension=[8],
    # The number of dynamic real valued features.
    # num_dynamic_real_features=len(train_dataset[0]['feat_dynamic_real']),
    # transformer params:
    encoder_layers=4,
    decoder_layers=4,
    d_model=32,
)

model = InformerForPrediction(config)

print("model created from with config: ", config)

print("creating train and test loaders")

train_dataloader = create_train_dataloader(
    config=config,
    freq=freq,
    data=train_dataset,
    batch_size=64,
    num_batches_per_epoch=100,
)

test_dataloader = create_test_dataloader(
    config=config,
    freq=freq,
    data=test_dataset,
    batch_size=64,
)

print("train loader features and dimensions")
batch = next(iter(train_dataloader))
for k, v in batch.items():
    print(k, v.shape, v.type())


accelerator = Accelerator()
device = accelerator.device

model.to(device)
optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

model, optimizer, train_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
)

print(f"starting to train model for {epochs} epochs")

model.train()

losses = []

for epoch in tqdm(range(epochs)):
    batch_losses = []
    for idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"]
            .type(torch.FloatTensor)
            .to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"]
            .type(torch.FloatTensor)
            .to(device),
            future_values=batch["future_values"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
            future_observed_mask=batch["future_observed_mask"].to(device),
        )
        loss = outputs.loss
        # Backpropagation
        accelerator.backward(loss)
        optimizer.step()

        batch_losses.append(loss.item())

    losses.append(sum(batch_losses) / len(batch_losses))


unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    f"./output/{training_parameters['name']}/roll-{rolling_window_size}-window-{window_size}-future-{prediction_length}/",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
)

print(
    f"model saved at: ./output/{training_parameters['name']}/Informer-Transformer-roll-{rolling_window_size}-window-{window_size}-future-{prediction_length}/ "
)
