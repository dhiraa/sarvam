# sarvam
Symbolizes the universal cosmic energy that resided in everything, living or non-living!


## About
This repo is a collection of Academic paper implementations covering some 
common use cases, using Tensorflow.

In general each paper implementaion is to be considered as individual mini-projects,
where it has its own utilities classes to handle data and visualizations.

**Structure**
Below simple naive approach is used for faster code developement and agility, with
less maintanace(we hope so ;)).
- model_name
    - README.md 
    - papers/ - Academic papers reffered
    - tmp/ - To store the preprocessed data and model files
    - run_*.py - encapsulated a general naive way of using the model APIs
             - all configs are present her for the model based on the dataset used

Also each model has its own global/local configs defined, which is not exposed via any API,
since the motive is to replicate the paper into code that can be easily extendable and not 
about the end use cases!

## Anaconda setup
- Refer https://www.anaconda.com/download/ for setup based on the OS you use
```bash

export PATH=/home/rpx/anaconda3/bin:$PATH

conda create -n sarvam python=3.5 anaconda
source activate sarvam
pip install tensorflow_gpu
pip install spacy
pip install tqdm

```

## OpenAI Gym setup
For those unfamiliar, the OpenAI gym provides an easy way for people to experiment 
with their learning agents in an array of provided toy games.
- https://github.com/openai/gym

## How to setup with IntelliJ
- File -> New Project and point to sarvam
- Select "sarvam" anaconda env as you project intrepretor, if not found 
continue with existing env and follow following step to switch to "sarvam" 
env or to any existing environment
- File -> Settings -> Project -> Project Interpretor -> settings symbol ->
    Add Local -> ~/anaconda3/env/sarvam
- In addition to that for each mini-project, click on root folder and do following
    - Right click -> Mark Directory As -> Sources Root
    
##Data

For Kaggle dataset we use Kaggle CLI from [here]()https://github.com/floydwch/kaggle-cli)

```
source activate sarvam
pip install kaggle-cli
```
    
    
## Text Classification
- [Fast Text](text_classification/fast_text)
- [CNN+RNN](text_classification/cnn_rnn)
