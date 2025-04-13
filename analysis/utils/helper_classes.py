import joblib
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
from sklearn.preprocessing import StandardScaler

class CancerDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        columns = df.columns.tolist()
        self.feature_names = columns[:-1]
        features = df.iloc[:, :-1].copy()
        target = df[columns[-1]].values
        
        self.x = torch.tensor(features.values, dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CancerDataModule(L.LightningDataModule):
    def __init__(self, df, numerical_features=None, random_state=42, scaler=None):
        super().__init__()
        self.random_state = random_state
        self.numerical_features = numerical_features
        self.df = df
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
    
    def save_scaler(self, path="results/scaler.pkl"):
        joblib.dump(self.scaler, path)

    def setup(self, stage=None):
        from sklearn.model_selection import train_test_split

        train_df, temp_df = train_test_split(self.df, test_size=0.4, random_state=self.random_state)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=self.random_state)
        if not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(train_df[self.numerical_features])
        train_df.loc[:, self.numerical_features] = self.scaler.transform(train_df[self.numerical_features])
        val_df.loc[:, self.numerical_features] = self.scaler.transform(val_df[self.numerical_features])
        test_df.loc[:, self.numerical_features] = self.scaler.transform(test_df[self.numerical_features])


        train_dataset = CancerDataset(train_df)

        self.ds_train = train_dataset
        self.ds_val = CancerDataset(val_df)
        self.ds_test = CancerDataset(test_df)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=32)
