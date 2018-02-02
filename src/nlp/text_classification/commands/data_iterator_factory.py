import sys
sys.path.append("../")

import tensorflow as tf
import argparse
from importlib import import_module

class DataIteratorFactory():

    iterator_path = {
        "text_char_ids": "nlp.text_classification.dataset.iterators.text_char_ids",
        "text_char_ids_ml": "nlp.text_classification.dataset.iterators.text_char_ids_ml",
        "text_ids": "nlp.text_classification.dataset.iterators.text_ids",
        "text_ids_lazy_generator" : "nlp.text_classification.dataset.iterators.text_ids_lazy_generator",
    }

    iterators = {
        "text_char_ids": "TextAndCharIds",
        "text_char_ids_ml": "TextAndCharIdsMultiLabel",
        "text_ids": "TextIds",
        "text_ids_lazy_generator" : "LazyGeneratorTextIds"
    }


    def __init__(self):
        pass

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


