import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .math import sparse_tensor_diag_matmul, sparse_scalar_multiply


def rescaled_laplacian(adj):
    """Creates a tensorflow (rescale) laplacian matrix out of a
    SparseTensorValue adjacency matrix."""

    degree = tf.sparse_reduce_sum(adj, axis=1)
    degree = tf.cast(degree, tf.float32)
    degree = tf.pow(degree, -0.5)
    lap = sparse_tensor_diag_matmul(adj, degree, transpose=True)
    lap = sparse_tensor_diag_matmul(lap, degree, transpose=False)

    return sparse_scalar_multiply(lap, -1)
