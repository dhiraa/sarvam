import sys
sys.path.append("../")

import tensorflow as tf
import argparse
from importlib import import_module

class ModelsFactory():

    model_path = {
        "fast_text_v0": "nlp.text_classification.models.fast_text.fast_text_v0",
        "bilstm_var_length_text" : "nlp.text_classification.models.bilstm.bilstm_var_length_text",
        "bilstm_multilabel": "nlp.text_classification.models.bilstm.bilstm_multilabel",
        "cnn_text_v0" : "nlp.text_classification.models.cnn.cnn_text_v0",
        "cnn_rnn_v0" : "nlp.text_classification.models.cnn.cnn_rnn_v0"
    }

    model_configurations = {
        "fast_text_v0": "FastTextV0Config",
        "bilstm_var_length_text": "BiLSTMVarTextConfig",
        "bilstm_multilabel": "BiLSTMMultiLabelConfig",
        "cnn_text_v0": "CNNTextV0Config",
        "cnn_rnn_v0": "MultiClassCNNRNNConfig"
    }


    models = {
        "fast_text_v0": "FastTextV0",
        "bilstm_var_length_text": "BiLSTMVarText",
        "bilstm_multilabel": "BiLSTMMultiLabelClassifier",
        "cnn_text_v0": "CNNTextV0",
        "cnn_rnn_v0": "MultiClassCNNRNNV0"
    }


    def __init__(self):
        pass

    @staticmethod
    def _get_model(name):

        try:
            model = getattr(import_module(ModelsFactory.model_path[name]), ModelsFactory.models[name])
        except KeyError:
            raise NotImplemented("Given config file name not found: {}".format(name))
        # Return the model class
        return model

    @staticmethod
    def _get_model_config(name):

        """
        Retrieves the model configuration, which later can be used to get user params
        """

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


