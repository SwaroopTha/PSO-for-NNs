from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, TensorDataset
import torch

def load_iris_data(batch_size=32):
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Normalize and convert to PyTorch tensors
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(dataset, batch_size=batch_size)

if __name__ == "__main__":
    # laod a sample
    data_loader = load_iris_data()
    for batch in data_loader:
        print(batch)
        print(batch[0].shape, batch[1].shape)
        break