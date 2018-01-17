import argparse
import sys
sys.path.append("../../")
sys.path.append("../")
sys.path.append(".")

import spacy
import tensorflow as tf
from tqdm import tqdm
nlp = spacy.load('en_core_web_sm')

from nlp.text_classification.commands.model_factory import ModelsFactory
from nlp.text_classification.commands.data_iterator_factory import DataIteratorFactory
from nlp.text_classification.commands.dataset_factory import DatasetFactory
from sarvam.helpers.print_helper import *


def run(opt):

    # Get the dataset
    dataset = DatasetFactory.get(opt.dataset_name)
    dataset = dataset()
    dataset.prepare()

    # Get the DataIterator
    data_iterator = DataIteratorFactory.get(opt.data_iterator_name)
    data_iterator = data_iterator(int(opt.batch_size), dataset.dataframe)
    # data_iterator_name.prepare()

    cfg, model = ModelsFactory.get(opt.model_name)

    # Get the model
    if opt.mode == "train":
        cfg = cfg.user_config(dataframe=dataset.dataframe, data_iterator_name=opt.data_iterator_name)
    elif opt.mode == "retrain" or opt.mode == "predict":
        cfg = cfg.load(opt.model_dir)

    model = model(cfg)


    if (model.feature_type != data_iterator.feature_type):
        raise Warning("Incompatible feature types between the model and data iterator")

    # Train and Evaluate
    batch_size = int(opt.batch_size)
    # NUM_STEPS = dataset.dataframe.num_train_samples // int(opt.batch_size)

    num_samples = dataset.dataframe.num_train_samples

    if (opt.mode == "train" or opt.mode == "retrain"):
        # Evaluate after each epoch
        for current_epoch in tqdm(range(int(opt.num_epochs))):
            tf.logging.info(CGREEN2 + str("Training epoch: " + str(current_epoch + 1)) + CEND)
            max_steps = (num_samples // batch_size) * (current_epoch + 1)

            model.train(input_fn=data_iterator.get_train_function(),
                        hooks=[data_iterator.get_train_hook()],
                        max_steps=max_steps)

            tf.logging.info(CGREEN2 + str("Evalution on epoch: " + str(current_epoch + 1)) + CEND)

            eval_results = model.evaluate(input_fn=data_iterator.get_val_function(),
                                          hooks=[data_iterator.get_val_hook()])

            tf.logging.info(CGREEN2 + str(str(eval_results)) + CEND)

    elif (opt.mode == "predict"):
        # Predict
        dataset.predict_on_csv_files(data_iterator, model)


if __name__ == "__main__":
    optparse = argparse.ArgumentParser("Run experiments on available models and datasets")


    optparse.add_argument('-mode', '--mode',
                          choices=['train', "retrain", "predict"],
                          required=True,
                          help="'preprocess, 'train', 'retrain','predict'"
                          )

    optparse.add_argument('-md', '--model-dir', action='store',
                          dest='model_dir', required=False,
                          help='Model directory needed for training')

    optparse.add_argument('-dsn', '--dataset-name', action='store',
                          dest='dataset_name', required=False,
                          help='Name of the Dataset to be used')

    optparse.add_argument('-din', '--data-iterator-name', action='store',
                          dest='data_iterator_name', required=False,
                          help='Name of the DataIterator to be used')

    optparse.add_argument('-bs', '--batch-size',  type=int, action='store',
                          dest='batch_size', required=False,
                          default=1,
                          help='Batch size for training, be consistent when retraining')

    optparse.add_argument('-ne', '--num-epochs', type=int, action='store',
                          dest='num_epochs', required=False,
                          help='Number of epochs')

    optparse.add_argument('-mn', '--model-name', action='store',
                          dest='model_name', required=False,
                          help='Name of the Model to be used')

    opt = optparse.parse_args()
    if (opt.mode == 'retrain' or opt.mode == 'predict') and not opt.model_dir:
        optparse.error('--model-dir argument is required in "retrain" & "predict" mode.')
    else:
        run(opt)
