# sarvam
Symbolizes the universal cosmic energy that resided in everything, living or non-living!


## About
This repo is a collection of Academic paper implementations covering some 
common use cases, using Tensorflow.

**No model is tuned for performance**, all we care is experimenting and 
lerning new deep learning models on existing dataset with minimum effort.

## Anaconda setup
- Refer https://www.anaconda.com/download/ for setup based on the OS you use
```bash

export PATH=/home/rpx/anaconda3/bin:$PATH

conda create -n sarvam python=3.5 anaconda
source activate sarvam
pip install tensorflow_gpu
pip install spacy
python -m spacy download en_core_web_lg
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

## [Natural Language Processing](src/nlp/)

### Tools:
- https://www.typora.io/

### References:
- https://devhints.io/jekyll