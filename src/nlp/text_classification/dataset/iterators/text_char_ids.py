import sys
sys.path.append("../")

import tensorflow as tf
from nlp.text_classification.dataset.data_iterator import DataIterator
from nlp.text_classification.tc_utils.feature_types import TextAndCharIdsFeature
from nlp.text_classification.tc_utils.tf_hooks.data_initializers import IteratorInitializerHook

class TextAndCharIds(DataIterator):
    def __init__(self, batch_size, dataframe):
        DataIterator.__init__(self)
        self.batch_size = batch_size
        self.dataframe = dataframe

        self.feature_type = TextAndCharIdsFeature

    def _setup_input_graph2(self,
                            word_ids,
                            char_ids,
                            labels,
                            batch_size,
                            is_eval=False,
                            shuffle=True,
                            scope='train_data'):
        """Return the input function to get the training audio_utils.

        Args:
            batch_size (int): Batch size of training iterator that is returned
                              by the input function.
            mnist_data (Object): Object holding the loaded mnist audio_utils.

        Returns:
            (Input function, IteratorInitializerHook):
                - Function that returns (features, labels) when called.
                - Hook to initialise input iterator.
        """
        iterator_initializer_hook = IteratorInitializerHook()

        tf.logging.info("text_features.shape: {}".format(word_ids.shape))
        tf.logging.info("numeric_features.shape: {}".format(char_ids.shape))
        tf.logging.info("labels.shape: {}".format(labels.shape))

        def inputs():
            """Returns training set as Operations.

            Returns:
                (features, labels) Operations that iterate over the dataset
                on every evaluation
            """
            with tf.name_scope(scope):

                # Define placeholders
                word_features_placeholder = tf.placeholder(tf.int32, word_ids.shape)
                char_features_placeholder = tf.placeholder(tf.int32, char_ids.shape)
                labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

                # Build dataset iterator
                dataset = tf.data.Dataset.from_tensor_slices(({self.feature_type.FEATURE_1: word_features_placeholder,
                                                               self.feature_type.FEATURE_2: char_features_placeholder},
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
                        feed_dict={word_features_placeholder: word_ids,
                                   char_features_placeholder: char_ids,
                                   labels_placeholder: labels})

                next_features, next_label = iterator.get_next()

                # Return batched (features, labels)
                return next_features, next_label

        # Return function and hook
        return inputs, iterator_initializer_hook

    def _test_inputs2(self, word_ids, char_ids, batch_size=1, scope='test_data'):
        """Returns test set as Operations.
        Returns:
            (features, ) Operations that iterate over the test set.
        """

        iterator_initializer_hook = IteratorInitializerHook()

        def inputs():
            with tf.name_scope(scope):
                word_features_placeholder = tf.placeholder(tf.int32, word_ids.shape)
                char_features_placeholder = tf.placeholder(tf.int32, char_ids.shape)
                labels_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                        name="labels") #dummy placeholder

                dataset = tf.data.Dataset.from_tensor_slices(
                    ({self.feature_type.FEATURE_1: word_features_placeholder,
                      self.feature_type.FEATURE_2: char_features_placeholder},
                     ))

                dataset = dataset.batch(batch_size)
                iterator = dataset.make_initializable_iterator()

                # Set runhook to initialize iterator
                iterator_initializer_hook.iterator_initializer_func = \
                    lambda sess: sess.run(
                        iterator.initializer,
                        feed_dict={word_features_placeholder: word_ids,
                                   char_features_placeholder: char_ids})

                next_features,  = iterator.get_next()

                next_label = None

                # Return batched (features, labels)
                return next_features, next_label


                # Return as iteration in batches of 1
                # return dataset.batch(batch_size).make_one_shot_iterator().get_next()

        return inputs, iterator_initializer_hook

    def prepare_train_set(self):
        '''
        Implement this function with reuqired TF function callbacks and hooks.
        :return:
        '''

        train_text_word_ids = self.dataframe.get_train_text_word_ids()
        train_text_word_char_ids = self.dataframe.get_train_text_word_char_ids()
        train_one_hot_encoded_label = self.dataframe.get_train_one_hot_label()

        self.train_input_fn, self.train_input_hook = self._setup_input_graph2(train_text_word_ids,
                                                                              train_text_word_char_ids,
                                                                              train_one_hot_encoded_label,
                                                                              self.batch_size,
                                                                              is_eval=False,
                                                                              shuffle=True,
                                                                              scope='train_data')


    def prepare_val_set(self):
        '''
        Implement this function with reuqired TF function callbacks and hooks.
        :return:
        '''

        val_text_word_ids = self.dataframe.get_val_text_word_ids()
        val_text_word_char_ids = self.dataframe.get_val_text_word_char_ids()
        val_one_hot_encoded_label = self.dataframe.get_val_one_hot_label()

        self.val_input_fn, self.val_input_hook = self._setup_input_graph2(val_text_word_ids,
                                                                          val_text_word_char_ids,
                                                                          val_one_hot_encoded_label,
                                                                          self.batch_size,
                                                                          is_eval=True,
                                                                          shuffle=True,
                                                                          scope='val_data')

    def prepare_test_set(self):
        '''
        Implement this function with reuqired TF function callbacks and hooks.
        :return:
        '''

        test_text_word_ids = self.dataframe.get_test_text_word_ids()
        test_text_word_char_ids = self.dataframe.get_test_text_word_char_ids()
        self.test_input_fn, self.test_input_hook = self._test_inputs2(test_text_word_ids,
                                           test_text_word_char_ids,
                                           batch_size=1,
                                           scope='test_data')

