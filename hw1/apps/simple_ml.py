import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl
import numpy as array_api
from needle.autograd import Tensor


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    def read_imgs(image_filename):
        data = gzip.open(image_filename, 'rb').read()
        # fmt of struct unpack, > means big endian, i means integer, well, iiii mean 4 integers
        fmt = '>iiii'
        offset = 0
        magic_number, img_number, height, width = struct.unpack_from(fmt, data, offset)
        offset += struct.calcsize(fmt)
        image_size = height * width  # 28x28
        fmt = '>{}B'.format(image_size)
        images = np.empty((img_number, height, width), dtype=np.float32)
        for i in range(img_number):
            images[i] = np.array(struct.unpack_from(fmt, data, offset)).reshape((height, width))
            offset += struct.calcsize(fmt)
        return images

    X = read_imgs(image_filename)

    def read_labels(label_filename):
        data = gzip.open(label_filename, 'rb').read()
        fmt = '>ii'
        offset = 0
        _, label_number = struct.unpack_from(fmt, data, offset)
        offset += struct.calcsize(fmt)
        fmt = '>B'
        labels = np.empty(label_number, dtype=np.uint8)
        for i in range(label_number):
            labels[i] = struct.unpack_from(fmt, data, offset)[0]
            offset += struct.calcsize(fmt)
        return labels

    y = read_labels(label_filename)

    def normalize(X):
        _max, _min = np.max(X), np.min(X)
        return (X - _min) / (_max - _min)

    normalized_X = normalize(X).reshape(-1, 28 * 28)
    return normalized_X, y
    ### END YOUR CODE



def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    batch_size, num_classes = Z.shape
    #y = Tensor(y.numpy().argmax(axis=1))
    #return 1 / batch_size * ndl.summation(ndl.log(ndl.summation(ndl.exp(Z), axes=1)) - Tensor(Z.numpy()[array_api.arange(batch_size), y]), axes=0)
    return 1 / batch_size * ndl.summation(ndl.log(ndl.summation(ndl.exp(Z), axes=1)) - ndl.summation(Z * y, axes=1))
    ### END YOUR CODE

def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    # def relu(x):
    #     return np.maximum(0, x)
    #
    def normalize(X: Tensor) -> Tensor:
        norms = ndl.broadcast_to(ndl.reshape(ndl.summation(X, axes=1), (X.shape[0], 1)), X.shape)
        return X / norms

    sample_num, num_classes = X.shape
    iter = int((sample_num + batch - 1) / batch)
    #iter = np.ceil(sample_num / batch).astype(np.int32)
    k = W2.shape[-1]
    for i in range(iter):
          l, r = i * batch, min((i + 1) * batch, sample_num)
          _X = Tensor(X[l:r, :])
          Z = ndl.relu(_X @ W1) @ W2
          y_one_hot = array_api.zeros_like(Z.numpy())
          y_one_hot[array_api.arange(Z.shape[0]), y[l:r]] = 1
          loss = softmax_loss(Z, Tensor(y_one_hot))
          loss.backward()
          W1 = Tensor(W1.numpy() - lr * W1.grad.numpy())
          W2 = Tensor(W2.numpy() - lr * W2.grad.numpy())
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
