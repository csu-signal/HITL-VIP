import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MarsDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        past_window_size: int = 50,
        future_time: int = 20,
        lstm_format=False,
        Xmeans: torch.Tensor = None,
        Xstds: torch.Tensor = None,
        Tmeans: torch.Tensor = None,
        Tstds: torch.Tensor = None,
    ) -> None:
        super().__init__()
        self.Xmeans = Xmeans
        self.Xstds = Xstds
        self.Tmeans = Tmeans
        self.Tstds = Tstds

        self.process_dataframe(
            dataframe, past_window_size=past_window_size, future_time=future_time
        )

        self.lstm_format = lstm_format

    def __len__(self):
        return len(self.features_physics)

    def __getitem__(self, idx):
        x = self.X[idx] if not self.lstm_format else self.X[idx].reshape(-1, 3)
        y = self.T[idx]
        return x.float(), y.float()

    def process_dataframe(
        self, dataframe: pd.DataFrame, past_window_size=50, future_time=1
    ) -> None:
        future_time = max(future_time, 1)

        features_physics = []
        metas = []
        targets = []
        trials = dataframe.peopleTrialKey.unique()

        def format_name(name: str):
            name = name[2:]
            name = name.replace("/", "")
            return name

        for trial in tqdm(trials):
            temp_data = dataframe[dataframe["peopleTrialKey"] == trial]
            temp_data = (
                temp_data[temp_data["trialPhase"].isin([3, 4])]
                .sort_values(["seconds"])
                .values
            )
            length_temp_data = len(temp_data)
            ppt_name = format_name(temp_data[0][5])
            for index, row in enumerate(temp_data):
                if row[1] == 3 and index + future_time < length_temp_data:
                    crash_in_near_future = (
                        4 in temp_data[index : index + future_time, 1]
                    )
                    if crash_in_near_future:
                        continue

                    if index - past_window_size < 0:
                        prev_n_values = temp_data[: index + 1, 2:5]
                    else:
                        prev_n_values = temp_data[
                            index - past_window_size : index + 1, 2:5
                        ]

                    prev_n_values = list(prev_n_values.reshape(-1))
                    future_joystick = temp_data[index + future_time, 4]

                    new_row = list(row[:2]) + [future_joystick] + prev_n_values
                    if len(prev_n_values) < ((past_window_size + 1) * 3):
                        diff = (past_window_size + 1) * 3 - len(prev_n_values)
                        prev_n_values = [0] * diff + prev_n_values

                    if len(prev_n_values):
                        metas.append(list(row[:2]) + list(row[5:7]))
                        features_physics.append(prev_n_values)
                        targets.append([future_joystick])

        self.features_physics = torch.tensor(features_physics, dtype=torch.float64)
        self.targets = torch.tensor(targets, dtype=torch.float64)
        self.metas = np.array(metas)

        if self.Xmeans is None:
            self.Xmeans = self.features_physics.mean(0).float()
            self.Xstds = self.features_physics.std(0).float()
            self.Xstds[self.Xstds == 0] = 1
            self.Tmeans = self.targets.mean(0).float()
            self.Tstds = self.targets.std(0).float()
            self.Tstds[self.Tstds == 0] = 1

        self.X = (self.features_physics - self.Xmeans) / self.Xstds
        self.T = (self.targets - self.Tmeans) / self.Tstds
