import tensorflow as tf
from tensorflow.python.platform import gfile
from tqdm import tqdm_notebook as tqdm
import tensorflow.contrib.learn as tflearn

from sarvam_utils.tf_hooks.data_initializers import IteratorInitializerHook
# Define data loaders
#https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html



def save_vocab(lines, outfilename, MAX_DOCUMENT_LENGTH, PADWORD='PADXYZ'):
    # the text to be classified
    vocab_processor = tflearn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH,
                                                                min_frequency=0)
    vocab_processor.fit(lines)

    with gfile.Open(outfilename, 'wb') as f:
        f.write("{}\n".format(PADWORD))
        for word, index in tqdm(vocab_processor.vocabulary_._mapping.items()):
            if len(word) > 0:
                f.write("{}\n".format(word))

    nwords = len(vocab_processor.vocabulary_) + 2  # PADWORD + UNKNOWN + ...
    print('{} words into {}'.format(nwords, outfilename))

    return nwords

# Define the inputs
def setup_input_graph(features, labels, batch_size, is_eval = False, shuffle=True, scope='train-data'):
    """Return the input function to get the training data.

    Args:
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
        mnist_data (Object): Object holding the loaded mnist data.

    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
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
            features_placeholder = tf.placeholder(tf.string, features.shape)
            labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

            # Build dataset iterator
            dataset = tf.contrib.data.Dataset.from_tensor_slices(({"text": features_placeholder},
                                                                  labels_placeholder))

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
                    feed_dict={features_placeholder: features,
                               labels_placeholder: labels})

            next_example, next_label = iterator.get_next()

            # Return batched (features, labels)
            return next_example, next_label

    # Return function and hook
    return inputs, iterator_initializer_hook

def test_inputs(features, batch_size=1, scope='test-data'):
    """Returns test set as Operations.
    Returns:
        (features, ) Operations that iterate over the test set.
    """
    def inputs():
        with tf.name_scope(scope):
            docs = tf.constant(features, dtype=tf.string)
            dataset = tf.contrib.data.Dataset.from_tensor_slices((docs,))
            # Return as iteration in batches of 1
            return dataset.batch(batch_size).make_one_shot_iterator().get_next()

    return inputs


def setup_input_graph2(text_features, numeric_features, labels, batch_size,
                       # num_epocs,
                       is_eval = False,
                       shuffle=True,
                       scope='train-data'):
    """Return the input function to get the training data.

    Args:
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
        mnist_data (Object): Object holding the loaded mnist data.

    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
    """
    iterator_initializer_hook = IteratorInitializerHook()

    tf.logging.info("text_features.shape: {}".format(text_features.shape))
    tf.logging.info("numeric_features.shape: {}".format(numeric_features.shape))
    tf.logging.info("labels.shape: {}".format(labels.shape))


    def inputs():
        """Returns training set as Operations.

        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        with tf.name_scope(scope):

            # Define placeholders
            text_features_placeholder = tf.placeholder(tf.string, text_features.shape)
            # numeric_featuress_placeholder = tf.placeholder(tf.float32, numeric_features.shape)
            labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

            # Build dataset iterator
            dataset = tf.contrib.data.Dataset.from_tensor_slices(({"text" : text_features_placeholder},
                                                                  # "numeric" :  numeric_featuress_placeholder},
                                                                  labels_placeholder))
            if is_eval:
                dataset = dataset.repeat(1)
            else:
                dataset = dataset.repeat(None)  # Infinite iterations

            if shuffle:
                dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={text_features_placeholder: text_features,
                               # numeric_featuress_placeholder: numeric_features,
                               labels_placeholder: labels})

            next_features, next_label = iterator.get_next()

            # Return batched (features, labels)
            return next_features, next_label

    # Return function and hook
    return inputs, iterator_initializer_hook