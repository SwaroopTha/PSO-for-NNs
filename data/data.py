from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import pandas as pd

def load_ucidata(name):
    """
    choices: iris, wine, breast_cancer
    """
    if name == 'iris':
        data = load_iris()
    elif name == 'wine':
        data = load_wine()
    elif name == 'breast_cancer':
        data = load_breast_cancer()
    elif name == 'digits':
        data = load_digits()
    elif name == 'diabetes':
        data = load_diabetes()
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    X, y = data.data, data.target
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    # loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = X.shape[1]
    output_dim = len(np.unique(y))

    return train_loader, test_loader, input_dim, output_dim
    

if __name__ == "__main__":
    # laod a sample
    train_loader, test_loader, input_dim, output_dim = load_ucidata("iris")
    print("input_dim:", input_dim)
    print("output_dim:", output_dim)
    for X, y in train_loader:
        print(X.shape, y.shape)
        break