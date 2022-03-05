
import tensorflow as tf
from tensorflow import keras

class BlockConv3D(keras.layers.Layer):
    def __init__(self, filters, *args, kernel_size=3, strides=1, padding='same', nconvs=1, normalization=True, activation=True, use_bias=False, name=None, kernel_initializer='he_normal', **kwargs):
        super().__init__(*args, name=name, **kwargs) # required
        self.normalization = normalization
        self.activation = activation

        self.nconvs = nconvs
        if self.normalization:
            self.bnorms = [
                keras.layers.BatchNormalization(
                    momentum=0.99,
                    name=None if name is None else name+'BatchNorm_'+str(i),
                )
                for i in range(self.nconvs)
            ]
        if self.activation:
            self.relu = keras.layers.ReLU()
        self.convs = [
            keras.layers.Conv3D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                name=None if name is None else name+'Conv3D_'+str(i),
                kernel_initializer=kernel_initializer,
            )
            for i in range(self.nconvs)
        ]

    def call(self, inputs, training=False, **kwargs):
        x = inputs
        for i in range(self.nconvs):
            if self.normalization:
                x = self.bnorms[i](x, training=training)
            if self.activation:
                x = self.relu(x)
            x = self.convs[i](x)
        return x

class ResBlockConv3D(keras.layers.Layer):
    """implements improved Residual block with full pre-activation from [1]. Allows better propagation of
    gradient signals unimpeded through very deep networks.

    [1] He, et al. Identity Mappings in Deep Residual Networks. arXiv preprint arXiv:1603.05027v3,2016
    """
    def __init__(self, filters, *args, kernel_size=3, strides=1, nconvs=1, normalization=True, activation=True, use_bias=False, name=None, kernel_initializer='he_normal', **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.normalization = normalization
        self.activation = activation

        self.nconvs = nconvs
        if self.normalization:
            self.bnorms = [
                keras.layers.BatchNormalization(
                    momentum=0.99,
                    name=None if name is None else name+'BatchNorm_'+str(i),
                )
                for i in range(self.nconvs)
            ]
        if self.activation:
            self.relu = keras.layers.ReLU()
        self.convs = [
            keras.layers.Conv3D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                activation=None,
                use_bias=False,
                name=None if name is None else name+'Conv3D_'+str(i),
                kernel_initializer=kernel_initializer,
            )
            for i in range(self.nconvs)
        ]

    def call(self, inputs, training=False, **kwargs):
        x = inputs
        for i in range(self.nconvs):
            if self.normalization:
                x = self.bnorms[i](x, training=training)
            if self.activation:
                x = self.relu(x)
            x = self.convs[i](x)
        # residual update
        output = x + inputs
        return output

class ResBlockConv3DBottleNeck(keras.layers.Layer):
    """implements improved Residual block with full pre-activation from [1]. Allows better propagation of
    gradient signals unimpeded through very deep networks.
    This is similar to the ResBlockConv2D class except that rather than stringing 'nconvs' identically sized conv2D ops with BN and activation between, this implements a bottleneck block which uses 1x1,3x3, and 1x1 convs in sequence to allow in_filters and out_filters to differ; it can change dimensionality of the feature map.

    [1] He, et al. Identity Mappings in Deep Residual Networks. arXiv preprint arXiv:1603.05027v3,2016
    """
    def __init__(self, in_filters, bottleneck_filters, kernel_size=3, strides=1, nconvs=1, name=None, kernel_initializer='he_normal'):
        super().__init__()

        self.relu = keras.layers.ReLU()

        name = (name+'/' if name else '')
        self.bottle_conv3d = keras.layers.Conv3D(
            filters=bottleneck_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            activation=None,
            use_bias=False,
            name=name+'Conv3D_bottle',
            kernel_initializer=kernel_initializer,
        )
        self.neck_conv3d = ResBlockConv3D(
            filters=bottleneck_filters,
            kernel_size=kernel_size,
            strides=strides,
            nconvs=nconvs,
            name=name+'Conv3D_neck',
            kernel_initializer=kernel_initializer,
        )
        self.unbottle_conv3d = keras.layers.Conv3D(
            filters=in_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            activation=None,
            use_bias=False,
            name=name+'Conv3D_unbottle',
            kernel_initializer=kernel_initializer,
        )

        self.bnorms = [
            keras.layers.BatchNormalization(
                momentum=0.99,
                name=name+'BatchNorm_'+str(i),
            )
            for i in range(2)
        ]

    def call(self, input, training=False, **kwargs):
        x = input # deepcopy of tensor
        x = self.bnorms[0](x, training=training)
        x = self.relu(x)
        x = self.bottle_conv3d(x)
        # resblock uses pre-activation so BN+activation is not necessary here
        x = self.neck_conv3d(x)
        x = self.bnorms[1](x, training=training)
        x = self.relu(x)
        # residual update
        output = input + x
        return output
