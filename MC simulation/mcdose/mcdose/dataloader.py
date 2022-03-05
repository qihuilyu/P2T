import logging
import ast
import os
import struct
import sys
import math
import random
import time
from os.path import join as pjoin
from collections import deque, namedtuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .timer import Timer

logger = logging.getLogger("MCDose."+__name__)

nnn = 0

def get_npy_files(folder, recursive=True):
    paths = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() == '.npy':
                paths.append(pjoin(root, f))
        if not recursive:
            del dirs[:]
    return paths

def load_data_from_file(file):
    trainarr = np.load(file)
    assert trainarr.ndim == 5
    labels = trainarr[...,0, None]
    inputs = trainarr[...,1:]
    return inputs, labels

def load_data_as_arrays(src_folder):
    lowvar = []
    highvar = []
    files = get_npy_files(pjoin(src_folder))

    for ii, npyfile in enumerate(files):
        logger.debug('loading data file {}/{}: "{!s}"'.format(ii+1, len(files), npyfile))

        h, l = load_data_from_file(npyfile)
        highvar.append(h)
        lowvar.append(l)

    if len(lowvar):
        lowvar = np.concatenate(lowvar)
        highvar = np.concatenate(highvar)

    return highvar, lowvar


DataPair = namedtuple("DataPair", ('inputs', 'labels'))
class LRUCache():
    """Least-recently-used cache with fixed capacity"""
    def __init__(self, capacity):
        self.capacity = capacity
        self._order = deque()
        self._data = {}

    def append(self, key, data):
        if key in self._data:
            # promote to newest and replace data
            self._order.remove(key)
            self._order.appendleft(key)
            self._data[key] = data
        else:
            if len(self._order) >= self.capacity:
                # remove least recently used data first
                lru = self._order.pop()
                del self._data[lru]

            # add data to cache and consider newest
            self._order.appendleft(key)
            self._data[key] = data
        #  logger.debug('lrucache size: {} of {}'.format(len(self), self.capacity))

    def remove(self, key):
        del self._data[key]
        self._order.remove(key)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.append(key, value)

    def __delitem__(self, key):
        self.remove(key)

    def __contains__(self, key):
        return key in self._data

    def get(self, key):
        return self._data.get(key)

class DataLoaderNpyFiles(keras.utils.Sequence):
    class Randomization():
        NoShuffle   = 0
        ShuffleOnce = 1
        Shuffle     = 2

    def __init__(self, files, batch_size, weight_func=None, full_batches_only=True, randomization=Randomization.Shuffle, cache_size=1, preload=False):
        """Creates an abstraction around a set of data chunks, organized as dim=4 tensors ([N,C,Y,X]) in .npy format

        By default, an LRU cache will be created that is capable of storing up to "cache_size" files at a time.
        When randomization==Randomization.Shuffle, both the file ordering and the data sample ordering within
        the active file will be shuffled. When one file is exhausted of all its data samples, the next file
        in the (possibly shuffled) list of files will be loaded (and its data samples possibly shuffled). If
        the file is in the LRUCache, it can be referenced rather than loaded at this time. If not, the least-
        recently referenced file is dropped from the cache, the file is loaded, and its (possibly shuffled)
        contents are placed in to the cache.
        """
        super().__init__()
        self.batch_size = batch_size
        self.weight_func = weight_func
        self.randomization = randomization
        self.files = list(set(files))
        self.file_cache = LRUCache(cache_size)

        if self.randomization == self.Randomization.ShuffleOnce and len(files) > cache_size:
            raise NotImplementedError(
                'In order to use "ShuffleOnce" randomization strategy, cache_size must be >= number of data files. \
                 Please increase cache_size or change randomization strategy.'
            )

        # get total number of examples, create map with map['<filename>'] = num_examples_in_file
        self.num_examples = 0
        self.num_examples_in_file = {}
        for f in self.files:
            shape = self.get_npy_array_shape(f)
            n = shape[0]

            self.num_examples_in_file[f] = n
            self.num_examples += n

        if self.num_examples <=0:
            raise RuntimeError("No data was loaded")

        self.num_batches = self.num_examples / self.batch_size  # only allow full batches
        if full_batches_only:
            self.num_batches = math.floor(self.num_batches)
        else:
            self.num_batches = math.ceil(self.num_batches)

        # Shuffle strategy will shuffle in .reset() instead
        if self.randomization == self.Randomization.ShuffleOnce:
            random.shuffle(self.files)
        self.reset()

        # preload first "cache_size" files
        if preload:
            for f in self.files:
                logger.info('Pre-loading file "{}"'.format(f))
                self.file_cache[f] = DataPair( *self.load_data(f) )

    def load_data(self, file):
        #  timer = Timer()
        #  timer.start('1. loading data')
        inputs, labels = load_data_from_file(file)
        if self.randomization >= self.Randomization.ShuffleOnce:
            # This creates a consistent shuffled ordering for inputs and labels
            # and creates a new view into existing array memory, so doesn't
            # consume any additional memory
            #  logger.debug(timer.restart_str('2. generating shuffled index'))
            shuf_idx = np.random.permutation(inputs.shape[0])
            #  logger.debug(timer.restart_str('3. permuting data'))
            inputs = inputs[shuf_idx]
            labels = labels[shuf_idx]
        #  logger.debug(timer.stop_str())
        return inputs, labels

    @classmethod
    def fromFolder(cls, folder, batch_size, *args, recursive=True, limit=None, **kwargs):
        files = sorted(get_npy_files(folder, recursive=recursive))
        if limit is not None and limit > 0:
            logger.info('Limiting dataset to {} files'.format(limit))
            files = files[:limit]
        self = cls(files, batch_size, *args, **kwargs)
        return self

    def get_batch(self, size):
        """get 'size' examples from 'file' beginning at 'offset', rollover
        to next file if not enough data left in current"""
        # ensure file_idx is valid
        if self.file_idx >= len(self.files):
            # rollover to new shuffling of files
            self.reset()

        file = self.files[self.file_idx]
        if file not in self.file_cache:
            inputs, labels = self.load_data(file)
            self.file_cache[file] = DataPair(inputs, labels)
        else:
            inputs, labels = self.file_cache[file]

        # get as many examples from this file as possible
        navailable = min(size, self.num_examples_in_file[file]-self.sample_idx)
        nremaining = size-navailable
        selector = slice(self.sample_idx, self.sample_idx+navailable)
        inputs = inputs[selector]
        labels = labels[selector]
        self.sample_idx += navailable

        if nremaining > 0:
            # we must have hit end of file, lets rollover, and get more data
            self.sample_idx = 0
            self.file_idx += 1
            # recursive call - when complete at base level, we have full batch
            _in, _la = self.get_batch(nremaining)
            inputs = np.concatenate([inputs, _in])
            labels = np.concatenate([labels, _la])

        inputs, labels = inputs.astype(np.float32), labels.astype(np.float32)

        #  global nnn
        #  import matplotlib.pyplot as plt
        #  for zz in range(9):
        #      plt.subplot(3,3,zz+1)
        #      plt.imshow(labels[0, zz, :,:,0])
        #  os.makedirs('/data/figures', exist_ok=True)
        #  plt.savefig('/data/figures/example{:05d}'.format(nnn))
        #  nnn+=1
        return inputs.astype(np.float32), labels.astype(np.float32)

    @staticmethod
    def get_npy_array_shape(f):
        with open(f, 'rb') as fd:
            if fd.read(6) != b'\x93NUMPY':
                raise RuntimeError('File "{}" is not a numpy "*.npy" file')
            vers_maj, vers_min = struct.unpack_from('BB', fd.read(2))
            hlen = struct.unpack_from('<H', fd.read(2))[0]
            head = ast.literal_eval(b''.join(struct.unpack_from('s'*hlen, fd.read(hlen))).decode('utf8'))
        return head['shape']

    def __getitem__(self, i):
        """get batch, argument 'i' is ignored since data is randomly sampled and trying to ensure deterministic
        retrieval is wasteful
        """
        inputs, labels = self.get_batch(self.batch_size)
        assert inputs.shape[0] > 0

        if self.weight_func is not None:
            weights = self.weight_func(inputs, labels)
            return inputs, labels, weights
        else:
            return inputs, labels

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        """allows class to function as an iterator as in 'for data in DataLoaderInstance' """
        batch_id = 0
        while batch_id < self.num_batches:
            yield self.__getitem__(None)
            batch_id += 1
        self.reset()

    def reset(self):
        self.file_idx = 0
        self.sample_idx = 0
        if self.randomization >= self.Randomization.Shuffle:
            random.shuffle(self.files)
