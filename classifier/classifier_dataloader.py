import numpy as np
import torch
from torch.utils.data import Dataset


class LatentsLabelDataset(Dataset):
    def __init__(self, file_path):
        """
        Args:
            file_path (str): Path to the dataset (npz file with "latents", "hints", "step_length", "begin_token_id").
        """
        self.data = np.load(file_path)
        self.latents = self.data['latents']
        self.hints = self.data['hints']
        self.step_length = self.data['step_length']
        self.begin_token_id = self.data['begin_token_id']
        self.correctness = self.data['correctness']
        self.logicality = self.data['logicality']
        
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, index):
        return {
            'latents': self.latents[index],
            'hints': self.hints[index],
            'step_length': self.step_length[index],
            'begin_token_id': self.begin_token_id[index],
            'correctness': self.correctness[index],
            'logicality': self.logicality[index]
        }
        
class CollateFn:
    def __init__(self):
        pass
    
    def __call__(self, batch):
        latents = torch.from_numpy(np.array([b['latents'] for b in batch])).float()
        hints = torch.from_numpy(np.array([b['hints'] for b in batch])).float()
        step_lengths = torch.tensor([b['step_length'] for b in batch], dtype=torch.float32).unsqueeze(1)
        begin_token_ids = torch.tensor([b['begin_token_id'] for b in batch], dtype=torch.float32).unsqueeze(1)
        correctness = torch.tensor([b['correctness'] for b in batch], dtype=torch.float32).unsqueeze(1)
        logicality = torch.tensor([b['logicality'] for b in batch], dtype=torch.float32).unsqueeze(1)
        return {'latents': latents, 
                'hints': hints, 
                'step_lengths': step_lengths, 
                'begin_token_ids': begin_token_ids, 
                'correctness': correctness,
                'logicality': logicality
                }
