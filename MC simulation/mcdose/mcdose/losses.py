import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger("MCDose."+__name__)

class MeanSquaredErrorTV(keras.losses.Loss):
    def __init__(self, *args, fidelity_cb=keras.losses.MeanSquaredError, tv_weight=0.2, compensated=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.tv_weight = tv_weight
        self.fidelity_cb = fidelity_cb(reduction=keras.losses.Reduction.NONE)

    def call(self, label, pred, weights=None):
        pred = tf.convert_to_tensor(pred)
        label = tf.cast(label, pred.dtype)

        fidelity = self.fidelity_cb(label, pred)
        if weights:
            fidelity *= weights
        fidelity = tf.reduce_mean(fidelity, axis=tf.range(1, tf.rank(fidelity)))
        tv = tf.image.total_variation(pred) / tf.cast(tf.reduce_prod(tf.shape(pred)[1:]), dtype=tf.float32)
        loss = fidelity + self.tv_weight*tv
        return loss


def meanabsoluteerror(labels, inputs):
    """standard L1 loss (MAE)"""
    return tf.math.abs(tf.math.subtract(inputs, labels))

def meansquarederror(labels, inputs):
    """standard L2 loss (MSE)"""
    return tf.math.pow(tf.math.subtract(inputs, labels), 2)

def uniform_weight(labels, inputs):
    """apply same weight to every voxel/example"""
    return tf.constant(1.0)

def _normalize(x, vmin=None, vmax=None):
    ax = tf.range(1,tf.rank(x))
    if vmax is None:
        vmax = tf.math.reduce_max(x, axis=ax, keepdims=True)
    if vmin is None:
        vmin = tf.math.reduce_min(x, axis=ax, keepdims=True)
    vrng = tf.subtract(vmax, vmin)
    return (x-vmin)/vrng

def linear_weight(labels, inputs):
    """apply linear weighting profile with weight=0.0 at inputs.min() and weight=1.0 at inputs.max()"""
    return _normalize(labels)

def exponential_weight(decay_rate=1.0):
    def f(labels, inputs):
        vmax = tf.math.reduce_max(labels, axis=tf.range(1,tf.rank(labels)), keepdims=True)
        nlabels = _normalize(labels, vmax=vmax)
        ninputs = _normalize(inputs, vmax=vmax)
        navg = (nlabels + ninputs) / 2.0
        weights = tf.math.minimum(1.0, tf.math.exp(-decay_rate*(1.0-navg)))
        return weights
    return f

def threshold_weight(thresh=0.1, mask_cval=0.0):
    """assign weight of 0 (ignore) to any voxels with normalized labels less than thresh"""
    def f(labels, inputs):
        vmax = tf.math.reduce_max(labels, axis=tf.range(1,tf.rank(labels)), keepdims=True)
        nlabels = _normalize(labels, vmax=vmax)
        ninputs = _normalize(inputs, vmax=vmax)
        weights = tf.where(tf.math.logical_or(nlabels>=thresh, ninputs>=thresh), 1.0, mask_cval)
        return weights
    return f



class WeightedLoss(keras.losses.Loss):
    def __init__(self, *args, loss_cb=meansquarederror, weight_cb=uniform_weight, **kwargs):
        super().__init__(*args, **kwargs, reduction=keras.losses.Reduction.NONE)

        self.weight_cb = weight_cb
        self.loss_cb   = loss_cb

    def call(self, labels, preds):
        weights = tf.cast(self.weight_cb(labels, preds), tf.float32)
        #  print('wts', weights.dtype, weights.shape)
        loss = tf.cast(self.loss_cb(labels, preds), tf.float32)
        #  print('loss', loss.dtype, loss.shape)
        wloss = tf.math.multiply(weights, loss)
        #  print(wloss.shape)
        rwloss = tf.reduce_sum(wloss) / tf.cast(tf.reduce_prod(tf.shape(loss)), tf.float32)
        return rwloss
