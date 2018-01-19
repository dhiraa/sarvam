# Introduction
An environment to try different models on different data sets, such that
the Tensorflow models(estimators) created can be reused out side of this
repo with ease!

# Related Work
- https://terrytangyuan.github.io/data/papers/tf-estimators-kdd-paper.pdf
- https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0 
- https://medium.com/towards-data-science/how-to-do-text-classification-using-tensorflow-word-embeddings-and-cnn-edae13b3e575    

**Git Repos**
- https://github.com/brightmart/text_classification

# Problem Statement
Keep data set, data iterators and models as independent as possible for
easy plug and play.

Any machine learning experiments has following steps:
- Data set preparation
- Data Preprocessing
- Data Iterartors
- Models
- Model Serving

Where **data preprocessing** is one time operation which inlcude and limited to 
- Cleaning the text
- Tokenning with NLP tools/libraries
- Padding the text

# Proposed Solution

# Validation


## Expected Data Format:

### Multiclass Classification

| text_col  | category_col |
|-----------|--------------|
| example_1 | class_1      |
| example_2 | class_3      |
| example_3 | class_2      |

Check the supported data set
 - [code](dataset/kaggle/spooky.py)
 - [Notebook](../../../data/notebooks/spooky_author_identification_dataset.ipynb)

### Multilabel Classification

| text_col  | category_col1 | category_col2 | category_col3 |
|-----------|---------------|---------------|---------------|
| example_1 | 0             | 1             | 0             |
| example_2 | 1             | 0             | 1             |
| example_3 | 0             | 1             | 1             |

Check the supported data set
 - [code](dataset/kaggle/jigsaw.py)
 - [Notebook](../../../data/notebooks/jigsaw_toxic_comment_classification_challenge_dataset.ipynb)

# sarvam/nlp/text_classification

## Directory setup
- experiments
    - data
        - dataset_1
        - dataset_2
    - models
        - dataset_1_name
            - dat_iterator_name
                - model_1
                    - config_1
                    - config_2
                - model_2
                    - config_1
                    - config_2

## Setup


## Plug and play Options
Run below command to get names of data sets, data iterators and
models that can run and tested effortlessly.


```
cd path/sarvam/src/

python nlp/text_classification/commands/get_command_options.py
```


## How to run?
 - [Fast Text](models/fast_text/)
 - [Convolutional Network](models/cnn/)
 - [Convolutional Network with RNN](models/cnn_rnn/)
 - [BiLSTM](models/bilstm/)