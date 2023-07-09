import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os
from pathlib import Path

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = []
    fn = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob), \
                       nn.Linear(hidden_dim, dim), norm(dim))
    modules.append(nn.Residual(fn))
    modules.append(nn.ReLU())
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, dim),
    )
    return nn.Sequential(modules)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt == None: model.eval()
    for idx, batch in enumerate(dataloader):
        batch_size = batch[0].shape[0]
        batch_x, batch_y = batch[0].reshape((batch_size, -1)), batch[1]
        pred = model(batch_x)
        print('test')
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    dim = 28 ** 2
    model = MLPResNet(dim, hidden_dim)
    data_path = Path(data_dir)
    mnist_train_dataset = ndl.data.MNISTDataset(data_path / "train-images-idx3-ubyte.gz",
                                                data_path / "train-labels-idx1-ubyte.gz")
    mnist_train_dataloader = ndl.data.DataLoader(dataset=mnist_train_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)
    mnist_test_dataset = ndl.data.MNISTDataset(data_path / "t10k-images-idx3-ubyte.gz",
                                               data_path / "t10k-labels-idx1-ubyte.gz")
    mnist_test_dataloader = ndl.data.DataLoader(dataset=mnist_test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)
    for _ in range(epochs):
        epoch(mnist_train_dataloader, model, optimizer)
        #epoch(mnist_test_dataloader, )
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
