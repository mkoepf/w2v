import torch
import numpy as np
import torch.nn as nn
from torch import Tensor, optim


def net(input_size: int, hidden_size: int, output_size: int):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Linear(hidden_size, output_size),
        nn.Softmax(1)
    )

    return model


def compute_loss(y_pred, y):
    return torch.nn.MSELoss(reduction='sum')(y_pred, y)


def train_model(model, x: np.array, y: np.array, epochs: int):
    learning_rate = 1e-5

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    x_tensor: Tensor = torch.from_numpy(x).float()
    y_tensor: Tensor = torch.from_numpy(y).float()

    for t in range(epochs):
        y_pred = model(x_tensor)

        loss: Tensor = compute_loss(y_pred.float(), y_tensor)
        if t % 10 == 9:
            print(t, loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
