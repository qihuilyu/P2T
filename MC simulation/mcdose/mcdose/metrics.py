import tensorflow as tf
from tensorflow import keras

def _normalize(x, vmin=None, vmax=None):
    ax = tf.range(1,tf.rank(x))
    if vmax is None:
        vmax = tf.math.reduce_max(x, axis=ax, keepdims=True)
    if vmin is None:
        vmin = tf.math.reduce_min(x, axis=ax, keepdims=True)
    vrng = tf.subtract(vmax, vmin)
    return (x-vmin)/vrng

def _normalize_to_ref(ref, arrs):
    vmax = tf.math.reduce_max(ref, axis=tf.range(1,tf.rank(ref)), keepdims=True)
    narrs = [_normalize(arr, vmin=0.0, vmax=vmax) for arr in arrs]
    return narrs

class MaskedNMAEMetric(keras.metrics.Metric):
    def __init__(self, threshold, name='masked_nmae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='nmae_sum', initializer='zeros', dtype=tf.float32)
        self.count = self.add_weight(name='count',   initializer='zeros', dtype=tf.int32)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        # SHAPE: [N, Z, Y, X, C]
        vmax = tf.math.reduce_max(y_true, axis=tf.range(1,tf.rank(y_true)), keepdims=True)
        nlabels = _normalize(y_true, vmin=0.0, vmax=vmax)
        npreds = _normalize(y_pred, vmin=0.0, vmax=vmax)

        absdiff = tf.math.abs(nlabels-npreds)
        predicate = tf.math.greater(nlabels, self.threshold)
        batch_masked_nmae = tf.math.reduce_sum(tf.where(predicate, absdiff, 0.0), axis=tf.range(1, tf.rank(absdiff))) / \
            tf.math.reduce_sum(tf.cast(predicate, tf.float32), axis=tf.range(1,tf.rank(absdiff)))
        self.total.assign_add(tf.math.reduce_sum(batch_masked_nmae))
        self.count.assign_add(tf.size(batch_masked_nmae))

    def result(self):
        return self.total / tf.cast(self.count, dtype=tf.float32)

class NMAEMetric(keras.metrics.Metric):
    def __init__(self, name='nmae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='nmae_sum', initializer='zeros', dtype=tf.float32)
        self.count = self.add_weight(name='count',   initializer='zeros', dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        vmax = tf.math.reduce_max(y_true, axis=tf.range(1,tf.rank(y_true)), keepdims=True)
        nlabels = _normalize(y_true, vmin=0.0, vmax=vmax)
        npreds = _normalize(y_pred, vmin=0.0, vmax=vmax)

        absdiff = tf.math.abs(nlabels-npreds)
        batch_nmae = tf.math.reduce_sum(absdiff, axis=tf.range(1, tf.rank(absdiff))) / tf.size(nlabels[0], out_type=tf.float32)
        self.total.assign_add(tf.reduce_sum(batch_nmae))
        self.count.assign_add(tf.size(batch_nmae))

    def result(self):
        return self.total / tf.cast(self.count, dtype=tf.float32)
