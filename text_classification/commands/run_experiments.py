import argparse
import sys
sys.path.append("../")

from commands.model_factory import ModelsFactory
from commands.data_iterator_factory import DataIteratorFactory
from commands.dataset_factory import DatasetFactory

def run(opt):

    # Get the dataset
    dataset = DatasetFactory(opt.dataset_name)

    # Get the DataIterator
    data_iterator = DataIteratorFactory(opt.data_iterator_name)
    data_iterator = data_iterator(opt.batch_size)
    data_iterator.prepare(dataset.dataframe)

    # Get the model
    cfg, model = ModelsFactory.get(opt.model_name)

    # Train and Evaluate

    # Predict


if __name__ == "__main__":
    optparse = argparse.ArgumentParser("Run experiments on available models and datasets")

    # CONLL specific preprocessing

    optparse.add_argument('-mode', '--mode',
                          choices=['preprocess', 'train', "retrain", "predict"],
                          required=True,
                          help="'preprocess, 'train', 'retrain','predict'"
                          )

    optparse.add_argument('-md', '--model-dir', action='store',
                          dest='model_dir', required=False,
                          help='Model directory needed for training')

    optparse.add_argument('-pd', '--predict-dir', action='store',
                          dest='predict_dir', required=False,
                          help='Model directory needed for prediction')

    optparse.add_argument('-model', '--model-name', action='store',
                          dest='model_name', required=False,
                          help='Model directory needed for training')

    opt = optparse.parse_args()
    if (opt.mode == 'retrain' or opt.mode == 'predict') and not opt.model_dir:
        optparse.error('--model-dir argument is required in "retrain" & "predict" mode.')

    elif opt.mode == 'predict' and not opt.predict_dir:
        optparse.error('--predict-dir argument is required in "predict" mode.')
    else:
        run(opt)
