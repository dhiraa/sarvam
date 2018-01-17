import pickle
import os

EXPERIMENT_ROOT_DIR = "experiments"
EXPERIMENT_DATA_ROOT_DIR = EXPERIMENT_ROOT_DIR + "/data/"
EXPERIMENT_MODEL_ROOT_DIR = EXPERIMENT_ROOT_DIR + "/models/"

class ModelConfigBase():
    @staticmethod
    def dump(model_dir, config):

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        with open(model_dir+"/model_config.pickle", "wb") as file:
            pickle.dump(config, file)

    @staticmethod
    def load(model_dir):
        with open(model_dir + "/model_config.pickle", "rb") as file:
            cfg = pickle.load(file)
        return cfg
