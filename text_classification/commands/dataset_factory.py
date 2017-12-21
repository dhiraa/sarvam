import sys
sys.path.append("../")

import tensorflow as tf
import argparse
from importlib import import_module

class DatasetFactory():

    dataset_path = {
        "spooky": "kaggle.spooky_dataset",
        "jigsaw": "kaggle.jigsaw_dataset"
    }

    models = {
        "spooky": "BiLSTMV0",
        "fast_text_v0" : "FastTextV0"
    }

    model_configurations = {
        "bilstm_v0": "BiLSTMConfigV0",
        "fast_text_v0": "FastTextV0Config"
    }

    def __init__(self):
        ""

    @staticmethod
    def _get_model(name):
        '''
        '''
        try:
            model = getattr(import_module(DatasetFactory.model_path[name]), DatasetFactory.models[name])
        except KeyError:
            raise NotImplemented("Given config file name not found: {}".format(name))
        # Return the model class
        return model

    @staticmethod
    def _get_model_config(name):
        '''
        '''
        try:
            cfg = getattr(import_module("models." + name), DatasetFactory.model_configurations[name])
        except KeyError:
            raise NotImplemented("Given config file name not found: {}".format(name))
        # Return the model class
        return cfg

    @staticmethod
    def get(model_name):
        cfg = DatasetFactory._get_model_config(model_name)
        model = DatasetFactory._get_model(model_name)
        return cfg, model


