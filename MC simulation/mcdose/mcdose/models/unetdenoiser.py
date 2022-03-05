import math
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .conv_blocks import BlockConv3D, ResBlockConv3D
from ..tf_functions import tf_normalize, tf_log

conv_initializer = 'he_normal'

logger = logging.getLogger('MCDose.'+__name__)


class UNetBranch(keras.layers.Layer):
    def __init__(self, filters, scales, *args, **kwargs):
        super().__init__(*args, **kwargs) # required

        self.scales = scales
        self.down_conv2d_ops = []
        self.down_scale_ops = []
        self.up_conv2d_ops = []
        self.up_scale_ops = []

        self.conv2d_block1 = keras.layers.Conv3D(filters=filters, kernel_size=5, padding='same', name=self.name+'Conv3D_block1', kernel_initializer=conv_initializer)

        nfilters = filters
        for scl in range(scales-1):
            name = self.name + '_down_s{:d}'.format(scl)
            self.down_scale_ops.append(
                #  keras.layers.AveragePooling3D(
                #      pool_size=(1,2,2),
                #      padding='same',
                #      name=name+"_maxpool2d"
                #  )
                keras.layers.Conv3D(
                    filters=nfilters,
                    kernel_size=3,
                    strides=(1,2,2),  # results in downsampling by 2x
                    padding='same',
                    activation=None,
                    use_bias=False,
                    name=name+'_stridedconv2d',
                    kernel_initializer=conv_initializer,
                )
            )
            nfilters *= 2
            self.down_conv2d_ops.append(BlockConv3D(
                filters=nfilters,
                kernel_size=3,
                nconvs=2,
                strides=1,
                padding='same',
                name=name+'_conv3d',
                kernel_initializer=conv_initializer,
            ))

        for scl in range(scales-1):
            name = self.name + '_up_s{:d}'.format(scl)
            nfilters //= 2
            self.up_scale_ops.append(keras.layers.Conv3DTranspose(
                filters=nfilters,
                kernel_size=3,
                strides=(1,2,2),  # results in upsampling by 2x
                padding='same',
                activation=None,
                use_bias=False,
                name=name+'_stridedconv3d',
                kernel_initializer=conv_initializer,
            ))
            self.up_conv2d_ops.append(BlockConv3D(
                filters=nfilters,
                kernel_size=3,
                nconvs=2,
                name=name+'_resconv3d',
                kernel_initializer=conv_initializer,
            ))

    def call(self, input, training=False, **kwargs):
        x = input

        # pad to make product of 2^n for some n (so downsampling always works)
        #  orig_size = tf.shape(input)[1:3]
        #  target_size = tf.cast(tf.math.pow(tf.constant([2, 2], tf.float32), tf.math.ceil(tf_log(tf.shape(x)[1:3], 2))), tf.int32)

        #  x = tf.image.pad_to_bounding_box(x, offset_height=0, offset_width=0,
        #                                   target_height=target_size[0],
        #                                   target_width=target_size[1])

        x = self.conv2d_block1(x)

        # down-scaling network
        skips = []
        for scl in range(self.scales-1):
            skips.append(x)
            x = self.down_scale_ops[scl](x)
            x = self.down_conv2d_ops[scl](x, training=training)

        # up-scaling network
        for scl in range(self.scales-1):
            x = self.up_scale_ops[scl](x)
            x = self.up_conv2d_ops[scl](x, training=training)
            x = tf.concat([x, skips[-1-scl]], axis=4)

        #  # crop to original input size
        #  x = tf.image.crop_to_bounding_box(x, offset_height=0, offset_width=0,
        #                                    target_height=orig_size[0],
        #                                    target_width=orig_size[1])

        return x

class UNetDenoiser(keras.Model):
    def __init__(self, *args, channel_groups=None, name='UNetDenoiser', **kwargs):
        super().__init__(*args, name=name, **kwargs) # required

        if channel_groups is None:
            channel_groups = [
                {'channels': [0],
                 'filters': 8,
                 'scales': 2,
                 }]

        logger.debug("UNetDenoiser Channel Groups: {!s}".format(channel_groups))
        channel_groups = self.convert_channel_map(channel_groups)
        self.branch_dose = UNetBranch(filters=32, scales=3)

        self.conv2d_mix = BlockConv3D(filters=32, kernel_size=1, nconvs=3, normalization=False, name='channel_mix', kernel_initializer=conv_initializer)
        self.conv2d_reduce = BlockConv3D(filters=1, kernel_size=1, normalization=False, name='channel_reduce', kernel_initializer=conv_initializer)

    def call(self, input, training=False, **kwargs):
        #  branch_results = []

        # push selected channels through corresponding branch
        #  for channels, branch in self.branches:
        #      branch_input = tf.stack([input[:,:,:,:,chan] for chan in channels], axis=4)
        #      branch_results.append(branch(branch_input, training=training))

        #  if len(branch_results)>1:
        #      x = tf.concat(branch_results, axis=4)
        #  else:
        #      x = branch_results[0]

        #  x = self.branch_dose(input, training=training)
        x = input

        # mix and reduce features
        x = self.conv2d_mix(x, training)
        x = self.conv2d_reduce(x)
        import matplotlib.pyplot as plt
        plt.subplot(1,3,1)
        plt.imshow(x.numpy()[0,5,:,:,0])
        plt.subplot(1,3,2)
        plt.imshow(input.numpy()[0,5,:,:,0])
        plt.subplot(1,3,2)
        plt.imshow(input.numpy()[0,5,:,:,1])
        plt.show()
        x = x + input[:,:,:,:,0, None]
        return x


    @staticmethod
    def convert_channel_map(channel_groups):
        channel_map = {'dose': 0, 'geometry': 1, 'fluence': 2}
        for channel_config in channel_groups:
            idx = []
            for c in list(channel_config['channels']):
                if isinstance(c, int) or str(c).isnumeric():
                    idx.append(int(c))
                elif c not in channel_map:
                    raise ValueError('Channel specification "{}" is not supported. Must be one of: {}'.format(c, list(channel_map.keys())))
                else:
                    idx.append(channel_map[c])
            channel_config['channels'] = sorted(idx)
        return channel_groups

def upsample_x2(nfilters):
    return keras.layers.Conv3DTranspose(
        filters=nfilters,
        kernel_size=3,
        strides=(1,2,2),  # results in upsampling by 2x
        padding='same',
        activation=None,
        use_bias=True,
        kernel_initializer=conv_initializer,
    )
def downsample_x2(nfilters):
    return keras.layers.Conv3D(
        filters=nfilters,
        kernel_size=3,
        strides=(1,2,2),  # results in downsampling by 2x
        padding='same',
        activation=None,
        use_bias=True,
        kernel_initializer=conv_initializer,
    )
def conv3d_3x3(nfilters, nconvs=1, normalization=True):
    return BlockConv3D(
        filters=nfilters,
        kernel_size=3,
        strides=1,
        padding='same',
        nconvs=nconvs,
        normalization=normalization,
        name=None,
        kernel_initializer=conv_initializer,
    )
def resconv3d_3x3(nfilters):
    return ResBlockConv3D(
        filters=nfilters,
        kernel_size=3,
        strides=1,
        nconvs=2,
        name=None,
        kernel_initializer=conv_initializer,
    )
def conv3d_1x1(nchannels,nconvs, norma):
    return BlockConv3D(
        filters=nchannels,
        kernel_size=1,
        kernel_initializer=conv_initializer)





class SEBlock(keras.Model):
    def __init__(self, nchan, *args, ratio=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.globavgpool = keras.layers.GlobalAveragePooling3D()
        self.dense_squeeze = keras.layers.Dense(nchan//ratio, activation='relu')
        self.dense_factor  = keras.layers.Dense(nchan, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.globavgpool(inputs)
        x = self.dense_squeeze(x)
        x = self.dense_factor(x)
        return keras.layers.multiply([inputs, x])

#  def seblock(inputs, nchan, ratio=16):
#      x = keras.layers.GlobalAveragePooling3D()(inputs)
#      x = keras.layers.Dense(nchan//ratio, activation='relu')(x)
#      x = keras.layers.Dense(nchan, activation='sigmoid')(x)
#      return keras.layers.multiply([inputs, x])

def recursive_unet(inputs, nfilters, nscales=1, post_down_block=None, post_cat_block=None, update_nfilters=None, training=False):
    """build a layer of unet and all sub-layers using recursive function call.

    nfilters = number of filters in first sub-scale, and at unet output
    update_nfilters: optional callable which matches signature:  new = func(old: int, is_down: bool) -> int
              the argument "is_down" indicates if this is immediately before a downsample op or immediately
              before an upsample op
    nscales=1 --> no downsampling/upsampling occurs
    post_down_block is an optional callable that is executed immediately after downsampling at each scale
    post_cat_block is an optional callable that is executed immediately after upsampling + skip connection
        concatenation at each scale

    all three block callables must conform to function signature:
        outputs = func(inputs: tensor, nfilters: int) -> tensor
    """
    # base case
    if nscales <=1:
        return inputs

    x = inputs

    # downsample
    if callable(update_nfilters):
        nfilters = update_nfilters(nfilters, is_down=True)
    x = keras.layers.Conv3D(
        filters=nfilters,
        kernel_size=(3,3,3),
        strides=(2,2,2),
        padding='same',
        use_bias=False,
        kernel_initializer=conv_initializer,
    )(x)

    # post_down_block
    if callable(post_down_block):
        x = post_down_block(x, nfilters)

    # next scale (recursive)
    x = recursive_unet(
        x, nfilters=nfilters,
        nscales=nscales-1,
        post_down_block=post_down_block,
        post_cat_block=post_cat_block,
        update_nfilters=update_nfilters,
        training=training,
    )

    # upsample
    if callable(update_nfilters):
        nfilters = update_nfilters(nfilters, is_down=False)
    x = keras.layers.Conv3DTranspose(
        filters=nfilters,
        kernel_size=(3,3,3),
        strides=(2,2,2),
        padding='same',
        use_bias=False,
        kernel_initializer=conv_initializer,
    )(x)
    # concatenate skip connection
    x = keras.layers.concatenate([x, inputs], axis=-1)

    # post_cat_block
    if callable(post_cat_block):
        x = post_cat_block(x, nfilters)

    return x

def UNetDenoiserStatic(inputs, training=False):
    nscales=3

    # unet block definition
    def seconv_block(_inputs, nfilters):
        x = _inputs
        x = BlockConv3D(
            filters=nfilters,
            kernel_size=1,
            strides=1,
            nconvs=1,
        )(x)
        x = ResBlockConv3D(
            filters=nfilters,
            kernel_size=(3,3,3),
            strides=1,
            nconvs=3,
            use_bias=False,
            kernel_initializer=conv_initializer,
        )(x)
        x = SEBlock(nfilters, 4)(x) # squeeze to 40//5 = 8 feature weights
        return x

    def update_nfilters(nfilters, is_down):
        return int(nfilters*(2 if is_down else 0.5))

    x = inputs

    # unet structure
    nfilters = 64
    x = recursive_unet(
        x,
        nfilters=nfilters,
        nscales=nscales,
        post_down_block=seconv_block,
        post_cat_block=seconv_block,
        update_nfilters=update_nfilters,
        training=training
    )

    # aggregate filters into final output (no )
    x = BlockConv3D(
        filters=1,
        kernel_size=1,
        strides=1,
        nconvs=1,
    )(x)
    x = ResBlockConv3D(
        filters=1,
        kernel_size=1,
        strides=1,
        nconvs=2,
    )(x)

    return x

def ModelWrapper(model, name=None):
    """Wrap Keras models that have been defined using eager operations api"""
    def wrap(*args, **kwargs):
        inputs = keras.Input(shape=(None, None, None, 2))
        return keras.Model(inputs, model(inputs), name=name)
    return wrap

def pad_for_unet(x, nscales):
    orig_size = tf.shape(x)[1:4]
    pads = tf.convert_to_tensor([[0, 0],
                                 [0, 2**(nscales-1)*(tf.math.floordiv(orig_size[0], 2**(nscales-1))+1) - orig_size[0]],
                                 [0, 2**(nscales-1)*(tf.math.floordiv(orig_size[1], 2**(nscales-1))+1) - orig_size[1]],
                                 [0, 2**(nscales-1)*(tf.math.floordiv(orig_size[2], 2**(nscales-1))+1) - orig_size[2]],
                                 [0, 0]], tf.int32)
    xp = tf.pad(x, pads)
    return xp, orig_size

def unpad_from_unet(x, orig_size):
    return x[:, :orig_size[0], :orig_size[1], :orig_size[2], :]


