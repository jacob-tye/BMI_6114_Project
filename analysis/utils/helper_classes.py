import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchmetrics import MeanSquaredError
from torchvision import datasets


class CancerDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        columns = df.columns.tolist()
        features = df.drop(columns=[columns[-1]])
        self.x = torch.tensor(features.values, dtype=torch.float32)
        self.y = torch.tensor(df[columns[-1]].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class CancerDataModule(L.LightningDataModule):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def setup(self, stage=None):
        self.ds_train,  self.ds_rest = random_split(
            self.dataset,
            [.6, .4]
        )

        self.ds_val, self.ds_test = random_split(
            self.ds_rest,
            [.5, .5]
        )
    
    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=32, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=32)
    
    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=32)
