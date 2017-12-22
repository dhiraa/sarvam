import sys
sys.path.append("../")

import tensorflow as tf
import argparse
from importlib import import_module

class ModelsFactory():

    model_path = {
        "bilstm_v0": "bilstm.bilstm_v0",
        "fast_text_v0": "fast_text.fast_text_v0"
    }

    models = {
        "bilstm_v0": "BiLSTMV0",
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
            model = getattr(import_module(ModelsFactory.model_path[name]), ModelsFactory.models[name])
        except KeyError:
            raise NotImplemented("Given config file name not found: {}".format(name))
        # Return the model class
        return model

    @staticmethod
    def _get_model_config(name):
        '''
        '''
        try:
            cfg = getattr(import_module(ModelsFactory.model_path[name]), ModelsFactory.model_configurations[name])
        except KeyError:
            raise NotImplemented("Given config file name not found: {}".format(name))
        # Return the model class
        return cfg

    @staticmethod
    def get(model_name):
        cfg = ModelsFactory._get_model_config(model_name)
        model = ModelsFactory._get_model(model_name)
        return cfg, model


