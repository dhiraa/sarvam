import sys
sys.path.append("../")

from importlib import import_module

class DatasetFactory():

    dataset_path = {
        "spooky": "nlp.text_classification.dataset.kaggle.spooky",
        "jigsaw": "nlp.text_classification.dataset.kaggle.jigsaw"
    }

    datasets = {
        "spooky": "SpookyDataset",
        "jigsaw" : "JigsawDataset"
    }

    def __init__(self):
        ""

    @staticmethod
    def _get_dataset(name):
        '''
        '''
        try:
            model = getattr(import_module(DatasetFactory.dataset_path[name]), DatasetFactory.datasets[name])
        except KeyError:
            raise NotImplemented("Given dataset file name not found: {}".format(name))
        # Return the model class
        return model

    @staticmethod
    def get(dataset_name):
        dataset = DatasetFactory._get_dataset(dataset_name)
        return dataset


