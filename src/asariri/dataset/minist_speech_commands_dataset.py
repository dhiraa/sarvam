import os

from asariri.dataset.crawled_data_interface import ICrawlingData


class CrawledData(ICrawlingData):
    def __init__(self, data_dir):
        ICrawlingData(data_dir=data_dir)

    def scan_directories(self):
        pass

    def get_train_files(self):
        return self._train_files

    def get_val_files(self):
        return self._val_files

    def get_test_files(self):
        return self._test_files