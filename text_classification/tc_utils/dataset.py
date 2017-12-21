import sys
sys.path.append("../")



class TextClassificationDataset():
    def __init__(self,
                 train_file_path,
                 test_file_path,
                 dataset_name):

        self.train_file_path = train_file_path
        self.test_file_path = test_file_path

        self.dataset_name =  dataset_name

        self.dataframe = None

    def prepare(self):
        raise NotImplementedError



