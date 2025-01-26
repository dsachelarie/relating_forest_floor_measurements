import numpy as np
import os.path
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from models.mlp import MLP
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


no_features = 10


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(np.array(self.X[idx]), dtype=torch.float32)
        y = torch.tensor(np.array(self.y[idx]), dtype=torch.float32)

        return X, y


if __name__ == "__main__":
    assert os.path.exists(f"datasets/{no_features}features_pca.csv") and os.path.exists("datasets/complete_fc.csv"), \
        "First run intermediate_points_generation.ipynb and feature_extraction.ipynb"

    X = pd.read_csv("datasets/10features_pca.csv")
    y = pd.read_csv("datasets/complete_fc.csv")
    y = y.drop(columns=["plot_ID", "location"]) * 100

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.1)

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(dataset=CustomDataset(X_train, y_train), batch_size=256, shuffle=True)
    test_loader = DataLoader(dataset=CustomDataset(X_test, y_test), batch_size=X_test.shape[0], shuffle=False)

    print("Training")

    for epoch in range(10000):
        model.train()
        loss_per_epoch = 0

        for (features, targets) in train_loader:
            optimizer.zero_grad()
            outputs = model.forward(features)
            loss = nn.functional.mse_loss(outputs, targets)
            loss_per_epoch += loss.item()

            loss.backward()
            optimizer.step()

        loss_per_epoch /= len(train_loader)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss {loss_per_epoch}")

    print("Testing")

    model.eval()

    with torch.no_grad():
        features, targets = next(iter(test_loader))
        y = model(features)
        print(targets)
        print(f"Errors: {np.abs(np.round(targets - y, decimals=1))}")
        print(f"MSE loss: {nn.functional.mse_loss(y, targets)}")
        print(f"MAE loss: {nn.functional.l1_loss(y, targets)}")
