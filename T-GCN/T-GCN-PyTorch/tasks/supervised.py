import argparse
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses


class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
        loss="huber",
        pre_len: int = 1,
        learning_rate: float = 0.0048,
        weight_decay: float = 7.5e-05,
        feat_max_val: float = 1.0,
        normalize_type="max",   # 🔥 NEW
        mean=0.0,               # 🔥 NEW
        std=1.0,                # 🔥 NEW
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self._loss = loss

        # 🔥 Normalization params
        self.normalize_type = normalize_type
        self.mean = mean
        self.std = std
        self.feat_max_val = feat_max_val

        # Regressor
        hidden_dim = (
            self.model.hyperparameters.get("hidden_dim")
            or self.model.hyperparameters.get("output_dim")
        )

        self.regressor = (
            nn.Linear(hidden_dim, self.hparams.pre_len)
            if regressor == "linear"
            else regressor
        )

    def forward(self, x):
        batch_size, _, num_nodes = x.size()

        hidden = self.model(x)  # (batch, nodes, hidden)
        hidden = hidden.reshape((-1, hidden.size(2)))

        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden

        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions

    def shared_step(self, batch):
        x, y = batch

        num_nodes = x.size(2)

        predictions = self(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y.reshape((-1, y.size(2)))

        return predictions, y

    # 🔥 FIXED LOSS
    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)

        if self._loss == "huber":
            return F.smooth_l1_loss(inputs, targets)

        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)

        raise ValueError(f"Loss not supported: {self._loss}")

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch)
        loss = self.loss(predictions, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _inverse_transform(self, predictions, y):
        if self.normalize_type == "zscore":
            predictions = predictions * self.std + self.mean
            y = y * self.std + self.mean
        else:
            predictions = predictions * self.feat_max_val
            y = y * self.feat_max_val
        return predictions, y

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch)

        # 🔥 Proper inverse normalization
        predictions, y = self._inverse_transform(predictions, y)

        loss = self.loss(predictions, y)

        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        explained_variance = utils.metrics.explained_variance(predictions, y)

        self.log_dict({
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            "accuracy": accuracy,
            "R2": r2,
            "ExplainedVar": explained_variance,
        }, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--learning_rate", "--lr", type=float, default=0.0048)
        parser.add_argument("--weight_decay", "--wd", type=float, default=7.5e-05)
        parser.add_argument("--loss", type=str, default="huber")

        # 🔥 NEW
        parser.add_argument("--normalize_type", type=str, default="max")

        return parser