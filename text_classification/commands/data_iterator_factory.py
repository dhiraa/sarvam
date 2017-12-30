import sys
sys.path.append("../")

import tensorflow as tf
import argparse
from importlib import import_module

class DataIteratorFactory():

    iterator_path = {
        "text_char_ids": "kaggle.data_iterators.text_char_ids",
        "text": "kaggle.data_iterators.text",
    }

    iterators = {
        "text_char_ids": "TextAndCharIds",
        "text": "TextIds",
    }


    def __init__(self):
        ""

    @staticmethod
    def _get_iterator(name):
        '''
        '''
        try:
            data_iterator = getattr(import_module(DataIteratorFactory.iterator_path[name]), DataIteratorFactory.iterators[name])
        except KeyError:
            raise NotImplemented("Given data iterator file name not found: {}".format(name))
        # Return the model class
        return data_iterator

    @staticmethod
    def get(iterator_name):
        iterator = DataIteratorFactory._get_iterator(iterator_name)
        return iterator


