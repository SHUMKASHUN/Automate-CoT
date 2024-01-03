import typing
from torch.utils.data import Dataset
import torch
import pandas as pd
import json


class KnowledgeDataset(Dataset):
    def __init__(
        self, data_path: str, size: typing.Optional[int] = None, *args, **kwargs
    ):
        print("Loading Knowledge dataset")
        
        with open(data_path, "r") as f:
            data = json.load(f)
        if size is not None:
            data = data[:size]
        self.data = self.preprocess(data)

        print(f"Loaded dataset with {len(self)} elements")
    
    def preprocess(self, data):
        return {
            'id': list(range(len(data))),
            'Question': [x['Question'] for x in data],
            'Rationale': [x['Rationale'] for x in data],
            'Answer': [x['Answer'] for x in data],
            'Ground_truth': [x['Ground_truth'] for x in data],           
        }

    def __len__(self):
        return len(self.data['id'])

    def __getitem__(self, item):
        return {
            'id': self.data['id'][item],
            'Question': self.data['Question'][item],
            'Rationale': self.data['Rationale'][item],
            'Answer': self.data['Answer'][item],
            'Ground_truth': self.data['Ground_truth'][item],
        }
        
    def collate_fn(self, batch):
        return {
            'id': [x['id'] for x in batch],
            'Question': [x['Question'] for x in batch],
            'Rationale': [x['Rationale'] for x in batch],
            'Answer': [x['Answer'] for x in batch],
            'Ground_truth': [x['Ground_truth'] for x in batch],
        }
