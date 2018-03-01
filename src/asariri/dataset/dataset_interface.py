import os


class IDataset:
    def __init__(self, data_dir):

        self.name = "none"

        self._data_dir = data_dir

        self.num_channels = -1
        self._train_files = []
        self._val_files = []
        self._test_files = []

        self.preprocess()

        self.NUM_TRAIN_SAMPLES = len(self._train_files)
        self.NUM_VAL_SAMPLES = len(self._val_files)

    def set_num_channels(self, num_channels):
        self.num_channels = num_channels

    def set_name(self, name):
        self.name = name

    def preprocess(self):
        raise NotImplementedError

    def get_train_files(self):
        return self._train_files

    def get_val_files(self):
        return self._val_files

    def get_test_files(self):
        return self._test_files