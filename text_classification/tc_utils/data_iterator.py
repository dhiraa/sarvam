import sys
sys.path.append("../")


class DataIterator():

    def __init__(self):

        self.feature_type = None

        self.train_input_fn = None
        self.train_input_hook = None

        self.val_input_fn = None
        self.val_input_hook = None

        self.test_input_fn = None
        self.test_input_hook = None

    def prepare(self, text_dataframe):
        '''
        Implement this function with reuqired TF function callbacks and hooks.
        Use one of the avaialble feature types in tc_utils.feature_types
        :return: 
        '''

        raise NotImplementedError

    def predict_on_csv_files(self, estimator):
        raise NotImplementedError

    def get_train_function(self):
        return self.train_input_fn

    def get_train_hook(self):
        return self.train_input_hook

    def get_test_function(self):
        return self.test_input_fn

    def get_test_hook(self):
        return self.test_input_hook

    def get_val_function(self):
        return self.val_input_fn

    def get_val_hook(self):
        return self.val_input_hook