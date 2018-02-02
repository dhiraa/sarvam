import sys
sys.path.append("../")

from importlib import import_module

class ModelsFactory():

    model_path = {
       "basic_model" : "asariri.models.basic_model"
    }

    model_configurations = {
        "basic_model": "BasicModelConfig"
    }


    models = {
        "basic_model" : "BasicModel"
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


