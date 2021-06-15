import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .layer import Layer
from .conv2d import Conv2d


class Fire(Layer):
    def __init__(self, in_channels, reduce_channels, out_channels, **kwargs):
        self._conv1 = Conv2d(in_channels, reduce_channels, size=1, **kwargs)
        self._conv2 = Conv2d(reduce_channels, out_channels, size=1, **kwargs)
        self._conv3 = Conv2d(reduce_channels, out_channels, size=3, **kwargs)

        super(Fire, self).__init__(**kwargs)

    def _call(self, inputs):
        outputs = self._conv1(inputs)
        outputs_1 = self._conv2(outputs)
        outputs_2 = self._conv3(outputs)
        return tf.concat([outputs_1, outputs_2], axis=3)
