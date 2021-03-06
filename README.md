# sarvam
Symbolizes the universal cosmic energy that resided in everything, living or non-living!


## About
This repo is a collection of Academic paper implementations covering some 
common use cases, using Tensorflow.

- [Natural Language Processing](src/nlp)
- [Speech Recognition](src/speech_recognition)

**No model is tuned for performance**, all we care is experimenting and 
learning new deep learning models on existing data set with minimum effort.

From the very fact that no single model can outperform any other models,
 this repo will act as a playground to learn and experiment Deep Learning models  

## Introduction
A simple and modular Tensorflow model development environment to handle variety of models.

Developing models to solve a problem for a data set at hand,
requires lot of trial and error methods.
Which includes and not limited to:
- Preparing the ground truth or data set for training and testing
    - Collecting the data from online or open data sources
    - Getting the data from in-house or client database
- Pre-processing the data set
    - Text cleaning
    - NLP processing
    - Meta feature extraction 
    - Audio pre-processing
    - Image resizing etc.,
- Data iterators, loading and looping the data examples for model
while training and testing
    - In memory - All data is held in RAM and looped in batches on demand
    - Reading from the disk on demand in batches
    - Maintaining different feature sets (i.e number of features and its types) for the model
- Models
    - Maintaining different models for same set of features
    - Good visualizing and debugging environment/tools
    - Start and pause the training at will
- Model Serving
    - Load a particular model from the pool of available models for a
    particular data set
    - Prepare the model for mobile devices
    
## Related Work
Most of the tutorials and examples out there for Tensorflow are biased for one data set or 
for one domain, which are rigid even if the tutorials are well written to handle same data sets.
In short we couldn't find any easy to experiment Tensorflow framework to play with different models. 

**We are happy to include if we find any such frameworks here in the future!**

## Problem Statement
 - To come up with an software architecture to try different models on
 different data set
 - Which should take care of:
    - Pre-processing the data
    - Preparing the data iterators for training, validation and testing
    for set of features and their types
    - Use a model that aligns with the data iterator and a feature type
    - Train the model in an iterative manner, with fail safe
    - Use the trained model to predict on new data
 - Keep the **model core logic independent** of the current architecture

# Solution or proposal

A few object-oriented principles are used in the python scripts for
ease of extensibility and maintenance.

## Current Architecture

- Handling Data set and Pre-processing
- Data iterators
    - Data set may have one or more features like words,
characters, positional information of words etc.,
    - Extract those and convert word/characters to numeric ids, pad them etc.,
    - Enforces number of features and their types, so that set of models
      can work on down the line
- Models should agree with data iterator features types and make use of the available features to train the data


![](docs/images/general_architecture.png)


- **Tensorflow Estimators** is used for training/evaluating/saving/restoring/predicting
   - [Official Guide](https://www.tensorflow.org/extend/estimators) 
   - [Tutorial](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html)

![](docs/images/tf_estimators.png)
 
## Anaconda setup
- Refer https://www.anaconda.com/download/ for setup based on the OS you use

## Requirements:
- Python 3.5
- tensorflow-gpu r1.4
- spaCy
- tqdm
- tmux
- overrides

```bash

export PATH=/home/rpx/anaconda3/bin:$PATH

conda create -n sarvam python=3.5 anaconda
source activate sarvam
pip install tensorflow_gpu
pip install spacy
python -m spacy download en_core_web_lg
pip install tqdm
pip install librosa
sudo apt-get install portaudio19-dev
```
## Git Clone
```commandline
git clone --recurse-submodules -j8 https://github.com/dhiraa/sarvam

#or if you wanted to pull after cloning

git submodule update --init --recursive

```
## How to setup with IntelliJ
- File -> New Project and point to sarvam
- Select "sarvam" anaconda env as you project intrepretor, if not found 
continue with existing env and follow following step to switch to "sarvam" 
env or to any existing environment
- File -> Settings -> Project -> Project Interpretor -> settings symbol ->
    Add Local -> ~/anaconda3/env/sarvam
- In addition to that for each mini-project, click on root folder and do following
    - Right click -> Mark Directory As -> Sources Root
    
## [Dataset](data)

## [Cheat Sheets](docs/cheat_sheets/) TODO

### Tools:
- https://www.typora.io/

### References:
- https://devhints.io/jekyll


### Never Ending References
- [150 ML Online matetials](https://unsupervisedmethods.com/over-150-of-the-best-machine-learning-nlp-and-python-tutorials-ive-found-ffce2939bd78)
