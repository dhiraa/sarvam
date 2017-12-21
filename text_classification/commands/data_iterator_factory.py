import sys
sys.path.append("../")

import tensorflow as tf
import argparse
from importlib import import_module

class DataIteratorFactory():

    iterator_path = {
        "text_char_ids": "kaggle.data_iterators.text_char_ids",
    }

    iterators = {
        "text_char_ids": "TextAndCharIds",
    }


    def __init__(self):
        ""

    @staticmethod
    def _get_iterator(name):
        '''
        '''
        try:
            model = getattr(import_module(DataIteratorFactory.iterator_path[name]), DataIteratorFactory.iterators[name])
        except KeyError:
            raise NotImplemented("Given config file name not found: {}".format(name))
        # Return the model class
        return model

    # @staticmethod
    # def _get_model_config(name):
    #     '''
    #     '''
    #     try:
    #         cfg = getattr(import_module("models." + name), DatasetFactory.model_configurations[name])
    #     except KeyError:
    #         raise NotImplemented("Given config file name not found: {}".format(name))
    #     # Return the model class
    #     return cfg

    @staticmethod
    def get(iterator_name):
        iterator = DataIteratorFactory._get_iterator(iterator_name)
        return iterator


