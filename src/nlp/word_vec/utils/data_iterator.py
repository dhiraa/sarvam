import tensorflow as tf
from word_vec.utils.tf_hooks.data_initializers import IteratorInitializerHook
import numpy as np

# Define the inputs
def glove_iterator(train_data, batch_size, shuffle=True, scope='train-data', is_eval=False):
    iterator_initializer_hook = IteratorInitializerHook()

    words = train_data["focal_words"]
    targets = train_data["context_words"]
    count = train_data["cooocurrence_count"]

    def inputs():
        """Returns training set as Operations.

        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        with tf.name_scope(scope):

            # Define placeholders
            words_placeholder = tf.placeholder(tf.string, words.shape, name="words")
            targets_placeholder = tf.placeholder(tf.string, targets.shape, name="targets")
            count_placeholder = tf.placeholder(tf.int32, count.shape, name="cooocurrence_count")

            # Build dataset iterator
            dataset = tf.contrib.data.Dataset.from_tensor_slices(({"words": words_placeholder,
                                                                   "targets": targets_placeholder,
                                                                   "count": count_placeholder}, None))

            if is_eval:
                dataset = dataset.repeat(1)  # Infinite iterations
            else:
                dataset = dataset.repeat(None)

            if shuffle:
                dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={words_placeholder: words,
                               targets_placeholder: targets,
                               count_placeholder: count})

            next_example, next_label = iterator.get_next()

            # Return batched (features, labels)
            return next_example, next_label

    # Return function and hook
    return inputs, iterator_initializer_hook

# Define the inputs
def skip_gram_iterator(train_data, batch_size, shuffle=False, scope='train-data', is_eval=False):
    """Return the input function to get the training/evaluation audio_utils.
    """
    iterator_initializer_hook = IteratorInitializerHook()

    def inputs():
        """Returns training set as Operations.

        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        with tf.name_scope(scope):

            # Define placeholders
            center_words_placeholder = tf.placeholder(tf.string, shape=[None,], name="center_words")
            target_words_placeholder = tf.placeholder(tf.string,  shape=[None,], name="target_words")

            # Build dataset iterator
            dataset = tf.data.Dataset.from_tensor_slices((center_words_placeholder,
                                                        target_words_placeholder))

            if is_eval:
                dataset = dataset.repeat(1)  # Infinite iterations
            else:
                dataset = dataset.repeat(None)

            if shuffle:
                dataset = dataset.shuffle(buffer_size=10000)

            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={center_words_placeholder: train_data["center_words"],
                               target_words_placeholder: train_data["target_words"]})

            next_example, next_label = iterator.get_next()

            # Return batched (features, labels)
            return next_example, next_label

    # Return function and hook
    return inputs, iterator_initializer_hook