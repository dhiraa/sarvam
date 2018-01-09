import argparse
import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")
import spacy
nlp = spacy.load('en_core_web_sm')

from commands.model_factory import ModelsFactory
from commands.data_iterator_factory import DataIteratorFactory
from commands.dataset_factory import DatasetFactory



def run(opt):

    # Get the dataset
    dataset = DatasetFactory.get(opt.dataset_name)
    dataset = dataset()
    dataset.prepare()

    # Get the DataIterator
    data_iterator = DataIteratorFactory.get(opt.data_iterator_name)
    data_iterator = data_iterator(int(opt.batch_size), dataset.dataframe)
    # data_iterator.prepare()

    cfg, model = ModelsFactory.get(opt.model_name)

    # Get the model
    if opt.mode == "train":
        cfg = cfg.user_config(dataframe=dataset.dataframe)
    elif opt.mode == "retrain" or opt.mode == "predict":
        cfg = cfg.load(opt.model_dir)

    model = model(cfg)


    if (model.feature_type != data_iterator.feature_type):
        raise Warning("Incompatible feature types between the model and data iterator")

    # Train and Evaluate
    NUM_STEPS = dataset.dataframe.num_train_samples // int(opt.batch_size)

    if (opt.mode == "train"):
        # Evaluate after each epoch
        for i in range(int(opt.num_epochs)):
            model.train(input_fn=data_iterator.get_train_function(),
                        hooks=[data_iterator.get_train_hook()],
                        steps=i + 1 * NUM_STEPS)

            model.evaluate(input_fn=data_iterator.get_val_function(), hooks=[data_iterator.get_val_hook()])
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

    optparse.add_argument('-pd', '--predict-dir', action='store',
                          dest='predict_dir', required=False,
                          help='Model directory needed for prediction')

    optparse.add_argument('-dsn', '--dataset-name', action='store',
                          dest='dataset_name', required=False,
                          help='Name of the Dataset to be used')

    optparse.add_argument('-din', '--data-iterator-name', action='store',
                          dest='data_iterator_name', required=False,
                          help='Name of the DataIterator to be used')

    optparse.add_argument('-bs', '--batch-size',  type=int, action='store',
                          dest='batch_size', required=False,
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

    elif opt.mode == 'predict' and not opt.predict_dir:
        optparse.error('--predict-dir argument is required in "predict" mode.')
    else:
        run(opt)
