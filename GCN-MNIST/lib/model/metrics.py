import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def softmax_cross_entropy(outputs, labels):
    """Calculate softmax cross-entropy loss."""

    loss_per_example = tf.nn.softmax_cross_entropy_with_logits(
        logits=outputs, labels=labels, name='loss_per_example')
    return tf.reduce_mean(loss_per_example, name='loss')


def total_loss(loss):
    tf.add_to_collection('losses', loss)
    losses = tf.get_collection('losses')
    return tf.add_n(losses, name='total_loss')


def accuracy(outputs, labels):
    """Calculate accuracy."""

    num_labels = labels.get_shape()[1]

    with tf.name_scope('accuracy'):
        labels = tf.cast(labels, tf.bool)

        predicted_labels = tf.argmax(outputs, axis=1)
        predicted_labels_one_hot = tf.one_hot(predicted_labels, num_labels)
        predicted_labels_one_hot = tf.cast(predicted_labels_one_hot, tf.bool)

        correct_prediction = tf.logical_and(predicted_labels_one_hot, labels)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        correct_prediction = tf.reduce_max(correct_prediction, axis=1)

        accuracy = tf.reduce_mean(correct_prediction)

    return accuracy


def precision(outputs, labels, k=0.5):
    with tf.name_scope('precision'):
        predicted_labels = _threshold_outputs(outputs, k)
        true_positives = _true_positives(labels, predicted_labels)
        true_and_false_positives = tf.reduce_sum(
            tf.cast(predicted_labels, tf.float32))

        res = true_positives / true_and_false_positives
        zero = tf.constant(0, tf.float32)
        return tf.cond(tf.is_nan(res), lambda: zero, lambda: res)


def recall(outputs, labels, k=0.5):
    with tf.name_scope('recall'):
        predicted_labels = _threshold_outputs(outputs, k)
        true_positives = _true_positives(labels, predicted_labels)
        relevant_elements = tf.reduce_sum(tf.cast(labels, tf.float32))

        res = true_positives / relevant_elements
        zero = tf.constant(0, tf.float32)
        return tf.cond(tf.is_nan(res), lambda: zero, lambda: res)


def _threshold_outputs(outputs, k=0.5):
    outputs = tf.nn.sigmoid(outputs)
    k = tf.zeros_like(outputs) + k
    outputs = tf.greater(outputs, k)
    return tf.cast(outputs, tf.uint8)


def _true_positives(labels, predicted_labels):
    labels = tf.cast(labels, tf.bool)
    predicted_labels = tf.cast(predicted_labels, tf.bool)

    true_positives = tf.logical_and(labels, predicted_labels)
    true_positives = tf.reduce_sum(tf.cast(true_positives, tf.float32))

    return true_positives
