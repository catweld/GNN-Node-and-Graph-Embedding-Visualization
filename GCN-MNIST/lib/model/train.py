from __future__ import print_function
from __future__ import division

import os
import time
from six.moves import xrange
from sklearn.preprocessing import StandardScaler

from ..datasets import PreprocessQueue
from .placeholder import feed_dict_with_batch
from ..pipeline import PreprocessedDataset, FileQueue, augment_batch


def train(model,
          data,
          preprocess_algorithm,
          batch_size,
          dropout,
          augment,
          max_steps,
          preprocess_first=None,
          display_step=10,
          save_step=250):

    train_queue, val_queue, test_queue = _generate_queues(
        data, preprocess_first, preprocess_algorithm, augment, batch_size)

    model.build()
    global_step = model.initialize()

    try:
        for step in xrange(global_step, max_steps):
            t_pre = time.process_time()
            batch = train_queue.dequeue()

            # Augment the preprocessed data.
            if augment and preprocess_first is not None:
                batch = augment_batch(batch) if augment else batch

            batch = _standard_scale_batch(batch)
            feed_dict = feed_dict_with_batch(model.placeholders, batch,
                                             dropout)
            t_pre = time.process_time() - t_pre

            t_train = model.train(feed_dict, step)

            if step % display_step == 0:
                # Evaluate on training and validation set with zero dropout.
                feed_dict.update({model.placeholders['dropout']: 0})
                batch = val_queue.dequeue()
                batch = _standard_scale_batch(batch)
                val_feed_dict = feed_dict_with_batch(model.placeholders, batch)

                train_info = model.evaluate(feed_dict, step, 'train')
                val_info = model.evaluate(val_feed_dict, step, 'val')

                log = 'step={}, '.format(step)
                log += 'time={:.2f}s + {:.2f}s, '.format(t_pre, t_train)
                log += 'train_loss={:.5f}, '.format(train_info[0])
                log += 'train_acc={:.5f}, '.format(train_info[1])
                log += 'val_loss={:.5f}, '.format(val_info[0])
                log += 'val_acc={:.5f}'.format(val_info[1])

                print(log)

            if step % save_step == 0:
                model.save()

    except KeyboardInterrupt:
        print()

    print('Optimization finished!')
    print('Evaluate on test set. This can take a few minutes.')

    try:
        num_steps = data.test.num_examples // batch_size
        test_info = [0, 0]

        for i in xrange(num_steps):
            batch = test_queue.dequeue()
            batch = _standard_scale_batch(batch)
            feed_dict = feed_dict_with_batch(model.placeholders, batch)

            batch_info = model.evaluate(feed_dict)
            test_info = [a + b for a, b in zip(test_info, batch_info)]

        log = 'Test results: '
        log += 'loss={:.5f}, '.format(test_info[0] / num_steps)
        log += 'acc={:.5f}'.format(test_info[1] / num_steps)

        print(log)

    except KeyboardInterrupt:
        print()
        print('Test evaluation aborted.')

    finally:
        train_queue.close()
        val_queue.close()
        test_queue.close()


def _preprocess_data(data, data_dir, preprocess_algorithm):
    data.train = PreprocessedDataset(
        os.path.join(data_dir, 'train'), data.train, preprocess_algorithm)
    data.val = PreprocessedDataset(
        os.path.join(data_dir, 'val'), data.val, preprocess_algorithm)
    data.test = PreprocessedDataset(
        os.path.join(data_dir, 'test'), data.test, preprocess_algorithm)
    return data


def _generate_queues(data, preprocess_first, preprocess_algorithm, augment,
                     batch_size):
    capacity = 10 * batch_size

    if preprocess_first is not None:
        data = _preprocess_data(data, preprocess_first, preprocess_algorithm)

        train_queue = FileQueue(data.train, batch_size, capacity, shuffle=True)
        val_queue = FileQueue(data.val, batch_size, capacity, shuffle=True)
        test_queue = FileQueue(data.test, batch_size, capacity, shuffle=False)
    else:
        train_queue = PreprocessQueue(
            data.train,
            preprocess_algorithm,
            augment,
            batch_size,
            capacity,
            shuffle=True)

        val_queue = PreprocessQueue(
            data.val,
            preprocess_algorithm,
            augment,
            batch_size,
            capacity,
            shuffle=True)

        test_queue = PreprocessQueue(
            data.test,
            preprocess_algorithm,
            augment,
            batch_size,
            capacity,
            shuffle=False)

    return train_queue, val_queue, test_queue


def _standard_scale_batch(batch):
    for example in batch:
        StandardScaler(copy=False).fit_transform(example[0])
    return batch
