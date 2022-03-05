import tensorflow as tf

def tf_normalize(ref, outlist):
    """normalize each array of outlist by the max value of ref per example
    This is equivalent to normalizing the dose of each beamlet to its own beamlet-specific maximum value"""
    if not isinstance(ref, tf.Tensor):
        ref = tf.convert_to_tensor(value=ref)
    tflist = []
    for x in outlist:
        if not isinstance(x, tf.Tensor):
            v = tf.convert_to_tensor(value=x)
        else: v = x
        tflist.append(v)
    norm_val =  tf.reduce_max(input_tensor=ref, axis=tf.range(1, tf.rank(ref)), keepdims=True)
    return [tf.truediv(x, norm_val) for x in tflist]


def image_neighbor_difference(images):
    ndims = images.get_shape().ndims

    if ndims == 3:
        # The input is a single image with shape [height, width, channels].

        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
        pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]

        pads1 = tf.constant([[0,0],[0,1],[0,0]])
        pads2 = tf.constant([[0,0],[0,0],[0,1]])
    elif ndims == 4:
        # The input is a batch of images with shape:
        # [batch, height, width, channels].

        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
        pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]

        pads1 = tf.constant([[0,0],[0,1],[0,0],[0,0]])
        pads2 = tf.constant([[0,0],[0,0],[0,1],[0,0]])
    else:
        raise ValueError('\'images\' must be either 3 or 4-dimensional.')

    pixel_dif1 = tf.pad(tensor=pixel_dif1, paddings=pads1)
    pixel_dif2 = tf.pad(tensor=pixel_dif2, paddings=pads2)
    wts = tf.abs(pixel_dif1) + tf.abs(pixel_dif2)
    return wts

def tf_log(x, b):
    """General log with base"""
    x = tf.cast(x, tf.float32)
    num = tf.math.log(x)
    return num / tf.math.log(tf.constant(b, dtype=num.dtype))
