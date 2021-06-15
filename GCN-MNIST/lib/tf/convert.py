import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def sparse_to_tensor(value):
    """Convert a scipy sparse matrix to a tensorflow SparseTensorValue."""

    row = np.reshape(value.row, (-1, 1))
    col = np.reshape(value.col, (-1, 1))
    indices = np.concatenate((row, col), axis=1)
    return tf.SparseTensorValue(indices, value.data, value.shape)
