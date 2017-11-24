import tensorflow as tf
from word_vec.utils.tf_hooks.data_initializers import IteratorInitializerHook

# Define the inputs
def setup_input_graph(center_words, target_words, batch_size, is_eval = False, shuffle=True, scope='train-utils'):
    """Return the input function to get the training/evaluation utils.
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
            center_words_placeholder = tf.placeholder(tf.string, center_words.shape, name="center_words")
            target_words_placeholder = tf.placeholder(tf.string, target_words.shape, name="target_words")

            # Build dataset iterator
            dataset = tf.contrib.data.Dataset.from_tensor_slices((center_words_placeholder,
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
                    feed_dict={center_words_placeholder: center_words,
                               target_words_placeholder: target_words})

            next_example, next_label = iterator.get_next()

            # Return batched (features, labels)
            return next_example, next_label

    # Return function and hook
    return inputs, iterator_initializer_hook