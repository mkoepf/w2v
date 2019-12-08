import torch
import numpy as np
import torch.nn as nn
from torch import Tensor


def net(input_size: int, hidden_size: int, output_size: int):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Linear(hidden_size, output_size),
        nn.Softmax(1)
    )

    return model


def compute_loss(y_pred, y):
    return torch.nn.MSELoss(reduction='sum')(y_pred, y)


def train_model(model, x: np.array, y: np.array):
    learning_rate = 1e-4

    for t in range(100):
        y_pred = model(torch.from_numpy(x).float())

        loss: Tensor = compute_loss(y_pred.float(), torch.from_numpy(y).float())
        if t % 10 == 9:
            print(t, loss.item())

        model.zero_grad()

        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
