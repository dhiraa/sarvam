import sys
sys.path.append("../")

from importlib import import_module

class ModelsFactory():

    model_path = {
        "simple_conv": "speech_recognition.models.conv.simple_conv",
        "cnn_trad_fpool3" : "speech_recognition.models.conv.cnn_trad_fpool3",
        "cnn_trad_fpool3_v1": "speech_recognition.models.conv.cnn_trad_fpool3_v1",
    }

    model_configurations = {
        "simple_conv": "SimpleSpeechRecognizerConfig",
        "cnn_trad_fpool3": "CNNTradFPoolConfig",
        "cnn_trad_fpool3_v1": "CNNTradFPoolConfigV1"
    }


    models = {
        "simple_conv": "SimpleSpeechRecognizer",
        "cnn_trad_fpool3": "CNNTradFPool",
        "cnn_trad_fpool3_v1": "CNNTradFPoolV1"
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


