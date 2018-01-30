import os


class ICrawlingData:
    def __init__(self, data_dir):
        self._data_dir = data_dir

        self._train_files = None # List of dictoinaries {"audio": "", "image":"", "label" : ""}
        self._val_files = None  # List of dictoinaries {"audio": "", "image":"", "label" : ""}
        self._test_files = None  # List of dictoinaries {"audio": "", "image":"", "label" : ""}
        pass

    def scan_directories(self):
        raise NotImplementedError

    def get_train_files(self):
        return self._train_files

    def get_val_files(self):
        return self._val_files

    def get_test_files(self):
        return self._test_files