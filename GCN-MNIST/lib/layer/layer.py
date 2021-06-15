import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

_LAYER_UIDS = {}


def _layer_uid(name):
    if name not in _LAYER_UIDS:
        _LAYER_UIDS[name] = 0

    _LAYER_UIDS[name] += 1
    return _LAYER_UIDS[name]


class Layer(object):
    def __init__(self, name=None, logging=False):

        if not name:
            layer = self.__class__.__name__.lower()
            name = '{}_{}'.format(layer, _layer_uid(layer))

        self.name = name
        self.logging = logging

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)

        return outputs

    def _call(self, inputs):
        return inputs
