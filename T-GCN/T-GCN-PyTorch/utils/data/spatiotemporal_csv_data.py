import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import utils.data.functions


class SpatioTemporalCSVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        feat_path,
        adj_path,
        batch_size=32,
        seq_len=14,
        pre_len=1,
        split_ratio=0.8,
        normalize=True,
        **kwargs
    ):
        super().__init__()

        self._feat_path = feat_path
        self._adj_path = adj_path

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize

        # Load data
        self._feat = utils.data.functions.load_features(self._feat_path)
        self._adj = utils.data.functions.load_adjacency_matrix(self._adj_path)

        # 🔥 Normalization stats
        self._mean = np.mean(self._feat)
        self._std = np.std(self._feat) + 1e-8
        self._feat_max_val = np.max(self._feat)

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--seq_len", type=int, default=14)
        parser.add_argument("--pre_len", type=int, default=1)
        parser.add_argument("--split_ratio", type=float, default=0.8)
        parser.add_argument("--normalize", type=bool, default=True)

        return parser

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset = utils.data.functions.generate_torch_datasets(
                self._feat,
                self.seq_len,
                self.pre_len,
                split_ratio=self.split_ratio,
                normalize=self.normalize,
            )

    # 🔥 CRITICAL: DataLoaders
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=0
        )

    # 🔥 Properties
    @property
    def adj(self):
        return self._adj

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std