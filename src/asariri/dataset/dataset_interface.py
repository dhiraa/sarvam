import os


class IDataset:
    def __init__(self, data_dir):
        self._data_dir = data_dir

        self._train_files = []
        self._val_files = []
        self._test_files = []

        self.preprocess()

        self.NUM_TRAIN_SAMPLES = len(self._train_files)
        self.NUM_VAL_SAMPLES = len(self._val_files)

    def preprocess(self):
        raise NotImplementedError

    def get_train_files(self):
        return self._train_files

    def get_val_files(self):
        return self._val_files

    def get_test_files(self):
        return self._test_files