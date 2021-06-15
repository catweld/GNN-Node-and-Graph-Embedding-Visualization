import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .var_layer import VarLayer


class FC(VarLayer):
    def __init__(self, in_channels, out_channels, dropout=None, **kwargs):
        self.dropout = dropout

        super(FC, self).__init__(
            weight_shape=[in_channels, out_channels],
            bias_shape=[out_channels],
            **kwargs)

    def _call(self, inputs):
        in_channels = self.vars['weights'].get_shape()[0].value

        outputs = tf.reshape(inputs, [-1, in_channels])

        if self.dropout is not None:
            outputs = tf.nn.dropout(outputs, 1 - self.dropout)

        outputs = tf.matmul(outputs, self.vars['weights'])

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return self.act(outputs)
