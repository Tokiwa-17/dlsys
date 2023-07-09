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
        nn.Linear(dim, hidden_dim), nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )
    return nn.Sequential(modules)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    training = True if opt else False
    if training: model.train()
    else: model.eval()
    loss_func = nn.SoftmaxLoss()
    loss, acc = [], 0.
    n_sample = 0

    for idx, batch in enumerate(dataloader):
        batch_x, batch_y = batch[0], batch[1]
        batch_x = batch_x.reshape((batch_x.shape[0], -1))
        n_sample += batch_x.shape[0]
        logits = model(batch_x)
        iter_loss = loss_func(logits, batch_y)
        #loss += iter_loss.cached_data
        loss.append(iter_loss.cached_data)
        pred = np.argmax(logits.cached_data, axis=1)
        acc += (pred == batch_y.cached_data).sum()
        if opt:
            iter_loss.backward()
            opt.step()
            opt.reset_grad()
    return 1 - acc / n_sample, np.mean(loss)
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
    for _epoch in range(epochs):
        train_loss, train_acc = epoch(mnist_train_dataloader, model, optimizer)
        test_loss, test_acc = epoch(mnist_test_dataloader, model, None)
        if _epoch == epochs - 1:
            return (train_acc, train_loss, test_acc, test_loss)
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
