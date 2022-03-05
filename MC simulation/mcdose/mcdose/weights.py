import tensorflow as tf

def sample_lin_norm_sum():
    """calculate per-sample weights to be used in training

    weight for each slice is normalized sum of label dose in slice for the batch
    so weight for the slice with highest sum of dose is 1.0, and other slices are in [0, 1.0]

    This is a factory function that should be instantiated without args before use for consistency with other
    weight functions in this module
    """
    @tf.function
    def func(inputs, labels):
        if not isinstance(labels, tf.Tensor):
            labels = tf.convert_to_tensor(labels, dtype=tf.float)
        sums = tf.math.reduce_sum(labels, axis=(1,2,3,4))
        max_sum = tf.math.reduce_max(sums)
        weights = sums / max_sum
        return tf.squeeze(weights)
    return func

def sample_exp_norm_sum(decay_rate):
    """Calculates per-sample weights for loss function evaluation

    Each sample is weighted according to $e^{a(x-1)}$ where $x$ is the sample (slice) sum of the label values
    normalized by the maximum sample sum of all samples in the batch, and a is a user-adjustable decay_rate parameter
    that adjusts the aggressiveness of the weight dropoff for less significant samples

    This is a factory function that should be instantiated for a specific decay_rate before use
    """
    _sample_lin_norm_sum = sample_lin_norm_sum() # instantiate now for reuse
    tfdecay_rate = tf.constant(decay_rate, dtype=tf.float32)
    @tf.function
    def func(inputs, labels):
        return tf.math.exp(tfdecay_rate * tf.cast(_sample_lin_norm_sum(inputs, labels) - 1.0, tf.float32))
    return func

def sample_thresh_norm_sum(decay_rate):
    """Calculates per-sample weights for loss function evaluation

    Each sample is weighted according to $e^{a(x-1)}$ where $x$ is the sample (slice) sum of the label values
    normalized by the maximum sample sum of all samples in the batch, and a is a user-adjustable decay_rate parameter
    that adjusts the aggressiveness of the weight dropoff for less significant samples

    This is a factory function that should be instantiated for a specific decay_rate before use
    """
    _sample_lin_norm_sum = sample_lin_norm_sum() # instantiate now for reuse
    tfdecay_rate = tf.constant(decay_rate, dtype=tf.float32)
    @tf.function
    def func(inputs, labels):
        return tf.math.exp(tfdecay_rate * tf.cast(_sample_lin_norm_sum(inputs, labels) - 1.0, tf.float32))
    return func

