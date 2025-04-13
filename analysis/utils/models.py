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


class CancerDataAutoEncoder(L.LightningModule):
    def __init__(self, input_size=20000, latent_size=200, n_layers=3, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = self._create_encoder(input_size, latent_size, n_layers, dropout)
        self.decoder = self._create_decoder(latent_size, input_size, n_layers, dropout)

        self.loss_metric = MeanSquaredError()
        self.val_metric = MeanSquaredError()
        self.test_metric = MeanSquaredError()


    def _create_encoder(self, input_size, latent_size, n_layers, dropout):
        layers = []
        current_size = input_size
        step = (input_size - latent_size) // n_layers

        for i in range(n_layers):
            next_size = max(latent_size, current_size - step)
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_size = next_size
        
        if current_size != latent_size:
            layers.append(nn.Linear(current_size, latent_size))

        return nn.Sequential(*layers)

    def _create_decoder(self, latent_size, output_size, n_layers, dropout):
        layers = []
        current_size = latent_size
        step = (output_size - latent_size) // n_layers

        for i in range(n_layers):
            next_size = min(output_size, current_size + step)
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_size = next_size

        if current_size != output_size:
            layers.append(nn.Linear(current_size, output_size))

        return nn.Sequential(*layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mse", self.loss_metric(x_hat, x), prog_bar=True)
        return loss

    def on_training_epoch_end(self):
        self.log("train_mse_epoch", self.loss_metric.compute())
        self.loss_metric.reset()

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse", self.val_metric(x_hat, x), prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_mse_epoch", self.val_metric.compute())
        self.val_metric.reset()

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_mse", self.test_metric(x_hat, x), prog_bar=True)
        return loss
    
    def on_test_epoch_end(self):
        self.log("test_mse", self.test_metric.compute())
        self.test_metric.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class BaseCancerRegressor(L.LightningModule):
    def __init__(self, auto_encoder, neural_network):
        super().__init__()
        self.auto_encoder = auto_encoder
        self.auto_encoder.requires_grad_(False)
        for param in self.auto_encoder.parameters():
            param.requires_grad = False
        self.neural_network = neural_network
        
        self.loss_metric = MeanSquaredError()
        self.val_metric = MeanSquaredError()
        self.test_metric = MeanSquaredError()


    def forward(self, x):
        x = self.auto_encoder.encoder(x)
        x = self.neural_network(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mse", self.loss_metric(y_hat, y), prog_bar=True)

        return loss
    
    def on_training_epoch_end(self):
        self.log("train_mse", self.loss_metric.compute())
        self.loss_metric.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse", self.val_metric(y_hat, y), prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_mse", self.val_metric.compute())
        self.val_metric.reset()
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.neural_network.parameters(), lr=1e-3, weight_decay=1e-4)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_mse", self.test_metric(y_hat, y), prog_bar=True)
        return loss
    
    def on_test_epoch_end(self):
        self.log("test_mse", self.test_metric.compute())
        self.test_metric.reset()