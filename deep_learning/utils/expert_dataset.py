import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class ExpertDataSet(Dataset):
    def __init__(self, expert_observations: np.ndarray, expert_actions: np.ndarray):
        self.observations = torch.tensor(expert_observations).float()
        self.actions = torch.tensor(expert_actions).float()

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)
