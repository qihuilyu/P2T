import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
pad_to_bounding_box = tf.image.pad_to_bounding_box
crop_to_bounding_box = tf.image.crop_to_bounding_box
from tensorflow_addons.image import transform
from tensorflow_addons.image.transform_ops import angles_to_projective_transforms

tf.get_logger().setLevel('INFO')

def tf_rotate(arr, angle):
    """take an array with shape [NDHWC], [NHWC], or [HW] and rotate each 2d slice of shape [HW]
    The resulting shape will match the input shape
    """
    orig_shape = arr.shape
    assert arr.ndim in (2, 4, 5)
    if arr.ndim == 5:
        arr = arr.reshape((-1, *arr.shape[2:]))

    if arr.ndim == 2:
        h, w = float(arr.shape[0]), float(arr.shape[1])
    elif arr.ndim == 4:
        h, w  = float(arr.shape[1]), float(arr.shape[2])

    proj = angles_to_projective_transforms(angle, h, w)
    trans = transform(arr, proj, interpolation='BILINEAR')
    trans = trans.numpy().reshape(orig_shape)
    return trans

def tf_padcrop(arr, target_height, target_width):
    """Either pad a small centered image with zeros to fill [target_height, target_width]
    or crop large centered image to match [target_height, target_width]
    """
    orig_shape = arr.shape
    assert arr.ndim in (2, 4, 5)
    if arr.ndim == 5:
        arr = arr.reshape((-1, *arr.shape[2:]))

    if arr.ndim == 2:
        h, w = arr.shape[0], arr.shape[1]
    elif arr.ndim == 4:
        h, w  = arr.shape[1], arr.shape[2]

    # pad, does nothing if already large enough
    th = max(h, target_height)
    tw = max(w, target_width)
    oh = max(0, (target_height-h)//2)
    ow = max(0, (target_width-w)//2)
    padarr = pad_to_bounding_box(tf.convert_to_tensor(arr), oh, ow, th, tw)

    # crop, does nothing if already small enough
    oh = max(0, (h-target_height)//2)
    ow = max(0, (w-target_width)//2)
    croppadarr = crop_to_bounding_box(padarr, oh, ow, target_height, target_width)
    croppadarr = croppadarr.numpy()
    if len(orig_shape) in (2, 4):
        return croppadarr
    elif len(orig_shape) == 5:
        return croppadarr.reshape((*orig_shape[0:2], *croppadarr.shape[2:]))

def tf_locbeamletcenter(arr):
    # sum of all values slice-wise. Slice with beamlet center should have largest sum
    # TODO: The assumption is not true. central slice doesn't always have largest sum
    slicewise_sum = tf.math.reduce_sum(arr, axis=range(1, arr.ndim))
    return np.argmax(slicewise_sum)
