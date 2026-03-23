import torch
import torch.nn.functional as F


def mse_with_regularizer_loss(inputs, targets, model, lamda=1e-4):
    return F.mse_loss(inputs, targets)