import numpy as np
import torch
from torch.utils.data import Dataset


class BaselinesActsDataset(Dataset):
    def __init__(self, file_path, input_key):
        """
        Args:
            file_path (str): Path to the dataset (.npz with layer_acts/sae_acts and correctness).
            input_key (str): Which input to use: "layer_acts" or "sae_acts".
        """
        data = np.load(file_path)
        if input_key not in data:
            raise KeyError(f"Missing key '{input_key}' in {file_path}")
        if "correctness" not in data:
            raise KeyError(f"Missing key 'correctness' in {file_path}")
        if "step_length" not in data:
            raise KeyError(f"Missing key 'step_length' in {file_path}")
        if "begin_token_id" not in data:
            raise KeyError(f"Missing key 'begin_token_id' in {file_path}")
        self.acts = data[input_key]
        self.correctness = data["correctness"]
        self.step_length = data["step_length"]
        self.begin_token_id = data["begin_token_id"]
        self.logicality = data.get("logicality", np.zeros_like(self.correctness))

    def __len__(self):
        return len(self.acts)

    def __getitem__(self, index):
        return {
            "acts": self.acts[index],
            "correctness": self.correctness[index],
            "step_length": self.step_length[index],
            "begin_token_id": self.begin_token_id[index],
            "logicality": self.logicality[index],
        }


class CollateFn:
    def __call__(self, batch):
        acts = torch.from_numpy(np.array([b["acts"] for b in batch])).float()
        correctness = torch.tensor(
            [b["correctness"] for b in batch], dtype=torch.float32
        ).unsqueeze(1)
        step_lengths = torch.tensor(
            [b["step_length"] for b in batch], dtype=torch.float32
        ).unsqueeze(1)
        begin_token_ids = torch.tensor(
            [b["begin_token_id"] for b in batch], dtype=torch.float32
        ).unsqueeze(1)
        logicality = torch.tensor(
            [b["logicality"] for b in batch], dtype=torch.float32
        ).unsqueeze(1)
        return {
            "acts": acts,
            "correctness": correctness,
            "step_lengths": step_lengths,
            "begin_token_ids": begin_token_ids,
            "logicality": logicality,
        }
