import numpy as np
import gzip
import struct
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return img[:, ::-1, :]
        else: return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        H, W, C = img.shape
        pad_img = np.pad(img, [[self.padding, self.padding], [self.padding, self.padding], [0, 0]], mode='constant', constant_values=0).astype(np.float64)
        x = self.padding + shift_x
        y = self.padding + shift_y
        crop = pad_img[x:x + H, y:y + W, :]
        return crop
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.idx = -1

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(self.dataset)),
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        else:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)),
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        self.idx += 1
        if self.idx >= len(self.ordering):
            self.idx = -1
            raise StopIteration
        return [Tensor(x) for x in self.dataset[self.ordering[self.idx]]]
        ### END YOUR SOLUTION

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

    #normalized_X = normalize(X).reshape(-1, 28 * 28)
    normalized_X = normalize(X).reshape(-1, 28, 28, 1)
    return normalized_X, y
    ### END YOUR CODE

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.X, self.y = parse_mnist(image_filename, label_filename)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        if self.transforms:
            return self.apply_transforms(self.X[index]), self.y[index]
        return self.X[index, :], self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
