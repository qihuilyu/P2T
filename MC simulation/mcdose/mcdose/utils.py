import gc
import os
from os.path import join as pjoin
import sys
from argparse import ArgumentTypeError
from pprint import pprint
import yaml
from datetime import datetime
import logging

import numpy as np
import numpy.random
import tensorflow as tf
from PIL import Image
import matplotlib as mpl
from matplotlib import colors
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.client import device_lib

logger = logging.getLogger("MCDose."+__name__)

def limited_float(low, high):
    """argparse string to float with requirement on allowable range of values"""
    def convert(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError("{} is not a floating point literal".format(x))
        if x < low or x > high:
            raise argparse.ArgumentTypeError("{} not in allowable range [{}, {}]".format(x, low, high))
        return x
    return convert

def randbool(p_true=0.5):
    p_true = max(0.0, min(1.0, p_true))
    return bool(np.random.rand()<p_true)

def augment_data(samples, labels):
    for mode in range(8):
        if randbool(0.2):
            samples = data_augmentation(samples, mode)
            labels = data_augmentation(labels, mode)
    return samples, labels


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        return np.array(im).reshape(1, im.size[1], im.size[0], 1)
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data

def load_bin(fname, size, add_channel_axis=False, norm=False):
    """loads binary double-precision data in ZYX order and returns an array in XZY order
    note: only works with 3d array"""
    with open(fname, 'rb') as fd:
        excepts = []
        for t in [np.float64, np.float32]:
            try:
                arr = np.frombuffer(fd.read(), dtype=t)
                arr = np.reshape(arr, size[::-1])
                break
            except Exception as e:
                excepts.append(e)

    if not (isinstance(arr, np.ndarray) and arr.size):
        raise RuntimeError(str('\n\n'.join([str(e) for e in excepts])))

    arr = arr.transpose(2,0,1).copy("C")
    if norm:
        arr /= np.max(arr)
    if add_channel_axis:
        arr = np.expand_dims(arr, axis=arr.ndim)
    return arr

def save_bin(fname, arr):
    """takes array in XZY order and saves to binary file as double-precision in ZYX order
    note: only works with 3d array"""
    arr = arr.transpose(1,2,0).copy("C")
    with open(fname, 'wb') as fd:
        fd.write(arr)

def save_as_image(filepath, arr, cmap=None, scale=1):
    _min = np.min(np.min(arr, axis=0), axis=0)
    _max = np.max(np.max(arr, axis=0), axis=0)
    outimg = (arr-_min)/(_max-_min)
    if cmap is not None:
        outimg = cmap(outimg)
    im = Image.fromarray(np.uint8(outimg*255))
    im = im.resize((scale*im.size[0], scale*im.size[1])) # make the image larger
    im.save(filepath)


def save_bin_slices(filepath, low_var, high_var, out_low_var, array_spacing=0):
    scale = 6
    with open(filepath+'.bin', 'wb') as f:
        output = np.zeros((low_var.shape[0], low_var.shape[1], 2))
        spacer = np.max(output)*np.ones((low_var.shape[0], low_var.shape[1], array_spacing))

        output = np.concatenate((output,high_var), axis=2)
        output = np.concatenate((output,spacer), axis=2)
        output = np.concatenate((output,low_var), axis=2)
        output = np.concatenate((output,spacer), axis=2)
        output = np.concatenate((output,out_low_var), axis=2)

        output = output.transpose(1,2,0)
        output = output.copy(order='C')
        f.write(output)  # save to bin

        save_as_image(filepath+'.png', output, cmap=mpl.cm.get_cmap('viridis'), scale=scale)


def tf_split_var(mode,images, percent, size, rank=4):
    k = tf.cast(tf.floor(tf.scalar_mul(percent, tf.cast(tf.size(input=images), tf.float32))), tf.int32)
    if mode =='low':
        values, idx = tf.nn.top_k(-images,k)
    elif mode == 'high':
        values, idx = tf.nn.top_k(images,k)

    mask = tf.compat.v1.sparse_to_dense(idx, tf.shape(input=images), sparse_values=0, default_value=1)
    images = tf.multiply(images,tf.cast(mask, tf.float32))
    if rank == 3:
      # The input is a single image with shape [height, width, channels].

      # Calculate the difference of neighboring pixel-values.
      # The images are shifted one pixel along the height and width by slicing.
      pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
      pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]

      # Sum for all axis. (None is an alias for all axis.)
      sum_axis = None
    elif rank == 4:
      # The input is a batch of images with shape:
      # [batch, height, width, channels].

      # Calculate the difference of neighboring pixel-values.
      # The images are shifted one pixel along the height and width by slicing.
      pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
      pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]

      # Only sum for the last 3 axis.
      # This results in a 1-D tensor with the total variation for each image.
      sum_axis = [1, 2, 3]
    else:
      raise ValueError('\'images\' must be either 3 or 4-dimensional.')

    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_var = (
        tf.reduce_sum(input_tensor=tf.abs(pixel_dif1), axis=sum_axis) +
        tf.reduce_sum(input_tensor=tf.abs(pixel_dif2), axis=sum_axis))

    return tot_var

class BatchConstructor():
    """Construct batches in sequence from a set of example (input, label) paired tensors """
    def __init__(self, inputs, labels):
        """ note: currently only handles 2D inputs. 3D extension should be simple

        Args:
            inputs (np.ndarray[N,H,W,C])
            labels (np.ndarray[N,H,W,C])
        """
        self.inputs = inputs
        self.labels = labels
        self.mark = 0

    def reset(self):
        """re-init the array of available example indices"""
        self.mark=0

    def iter_batches(self, batch_size):
        """construct batch in NHWC format where the first axis (N) is constucted of examples drawn from list of unused examples"""
        remain = len(self.labels) - self.mark
        mark = 0
        while remain > 0:
            _batch_size = min(remain, batch_size)
            yield (self.inputs[mark:mark+_batch_size,], self.labels[mark:mark+_batch_size,])
            remain -= _batch_size
            mark += _batch_size

class RandomBatchConstructor():
    """Construct batches randomly from a set of example (input, label) paired tensors """
    def __init__(self, inputs, labels):
        """ note: currently only handles 2D inputs. 3D extension should be simple

        Args:
            inputs (np.ndarray[N,H,W,C])
            labels (np.ndarray[N,H,W,C])
        """
        self.inputs = inputs
        self.labels = labels
        self.randorder = np.arange(self.inputs.shape[0])
        self.mark = 0
        self.initialized = False # lazy loading of index

    def reset(self):
        """re-init the array of available example indices"""
        self.initialized = True
        np.random.shuffle(self.randorder)
        self.mark = 0

    def make_batch(self, batch_size):
        """construct batch in NHWC format where the first axis (N) is constucted of random examples drawn from list of unused examples"""
        if not self.initialized:
            self.reset()
        remaining = len(self.randorder) - self.mark
        _batch_size = min(remaining, batch_size)
        if remaining < _batch_size:
            raise RuntimeError('There are not enough examples ({:d}) to fill the requested batch ({:d})'.format(remaining, _batch_size))
        selection = self.randorder[self.mark:self.mark+_batch_size]
        self.mark += _batch_size
        return (self.inputs[selection,], self.labels[selection,])

class RandomBatchConstructor_MultiTensor():
    """Construct batches randomly from a set of example (input, label) paired tensors. This class differs from
    RandomBatchConstructor in that it handles datasets consisting of more than one tensor of different dims"""
    def __init__(self, inputs, labels):
        """ note: currently only handles 2D inputs. 3D extension should be simple

        Args:
            inputs ([np.ndarray[N,H,W,C], ...])
            labels ([np.ndarray[N,H,W,C], ...])
        """
        self.inputs = inputs
        self.labels = labels
        self.index = [[]]*len(inputs)
        self.initialized = False # lazy loading of index

    def reset(self):
        """re-init the array of available example indices"""
        self.initialized = True
        self.index = []
        for input in self.inputs:
            self.index.append( list(range(input.shape[0])) )

    def make_batch(self, batch_size, reuse=False):
        """construct batch in NHWC format where the first axis (N) is constructed of random examples drawn from list of unused examples"""
        if not self.initialized:
            self.reset()

        # randomly select a dataset tensor
        unseen = list(range(len(self.inputs)))
        while True:
            if not len(unseen):
                raise RuntimeError('There are no remaining examples from which to fill the requested batch')
            tidx = np.random.choice(unseen)
            if len(self.index[tidx]):
                break
            unseen.remove(tidx)

        _batch_size = min(len(self.index[tidx]), batch_size)
        if len(self.index[tidx]) < _batch_size:
            raise RuntimeError('There are not enough examples ({:d}) to fill the requested batch ({:d})'.format(len(self.index[tidx]), _batch_size))
        select = np.random.choice(range(len(self.index[tidx])), _batch_size, replace=False)
        if not reuse:
            for idx in sorted(select, reverse=True):
                del self.index[tidx][idx]
        return (self.inputs[tidx][select,], self.labels[tidx][select,])


def get_available_gpus():
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.01, allow_growth=True))) as sess:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
        local_device_protos = device_lib.list_local_devices()

        devs =  [x.name for x in local_device_protos if x.device_type=='GPU']
    return devs

def get_unique_run_name(base, timestamp=False):
    """get unique directory name with lowest integer prefix"""
    runs = [int(x.split('-')[0]) for x in os.listdir(base) if os.path.isdir(pjoin(base, x))]
    runid = max(runs)+1 if len(runs) else 1
    runname = '{:04d}'.format(runid)
    if timestamp:
        timestamp = datetime.strftime('%Y%m%d-%H%M%S')
        runname += '-' + timestamp
    return runname

def save_config(dest, config):
    assert os.path.isdir(dest)
    dest = pjoin(dest, 'config.yml')
    with open(dest, 'w') as fd:
        yaml.safe_dump(config, fd)

def load_config(src):
    if os.path.isdir(src):
        src = pjoin(src, 'config.yml')
    with open(src, 'r') as fd:
        config = yaml.safe_load(fd)
    logger.info('Config loaded from "{}"'.format(src))
    return config

def combine_config(c1, c2):
    d = c1.copy()
    for k, v in c2.items():
        if k not in d or d[k] is None:
            d[k] = v
    return d
