# Introduction
Create an software environment/architecture to try different software models using
Tensorflow, Scikit and Keras framework on different datasets.
Come up with ensemble pipeline that uses these models on a dataset with ease.

# Related Work
- https://terrytangyuan.github.io/data/papers/tf-estimators-kdd-paper.pdf
- https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0 
- https://medium.com/towards-data-science/how-to-do-text-classification-using-tensorflow-word-embeddings-and-cnn-edae13b3e575    
- Inspired after: https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn

# Problem Statement
Keep dataset, data iterators and models as independent as possible for easy plug and play.

# Proposed Solution

# Validation


# sarvam/text_classification

## Directory setup
 - model_name
    - model_v0
    - model_v1
 - *_dataset.ipynb : Some exploratory data analysis on supported dataseets
 - experiments
    - data
        - dataset_1
        - dataset_2
    - models
        - dataset_1
            - model_1
            - model_2
        - dataset_2
            - model_1
## Setup


## Avaiable Models

## How to run?

**Tensorflow Models**
```
python commands/run_experiments.py \
--mode=train \
--dataset-name=spooky \
--data-iterator-name=text_char_ids \
--batch-size=32 \
--num-epochs=5 \
--model-name=bilstm_v0
```


```
python commands/run_experiments.py \
--mode=retrain \
--dataset-name=spooky \
--data-iterator-name=text_char_ids \
--batch-size=32 \
--num-epochs=5 \
--model-name=bilstm_v0 \
--model-dir=
```