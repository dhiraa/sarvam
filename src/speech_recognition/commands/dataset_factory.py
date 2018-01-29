import sys
sys.path.append("../")

from importlib import import_module

class DatasetFactory():

    dataset_path = {
        "speech_commands_v0": "speech_recognition.dataset.speech_commands_v0",
        "tensorflow_dataset_kaggle" : "speech_recognition.dataset.tensorflow_dataset_kaggle",
    }

    datasets = {
        "speech_commands_v0": "SpeechCommandsV0",
        "tensorflow_dataset_kaggle" : "KaggleDS"
    }

    def __init__(self):
        pass

    @staticmethod
    def _get_dataset(name):
        try:
            dataset = getattr(import_module(DatasetFactory.dataset_path[name]), DatasetFactory.datasets[name])
        except KeyError:
            raise NotImplemented("Given dataset file name not found: {}".format(name))
        # Return the model class
        return dataset

    @staticmethod
    def get(dataset_name):
        dataset = DatasetFactory._get_dataset(dataset_name)
        return dataset


