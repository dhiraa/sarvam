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


### Tools:
- https://www.typora.io/

### References:
- https://devhints.io/jekyll

-------------------------------------------------------------------------------------------------------
**Sub modules notes:**

```commandline
#add submodule and define the master branch as the one you want to track  
git submodule add -b master [URL to Git repo]     
git submodule init

#update your submodule --remote fetches new commits in the submodules 
# and updates the working tree to the commit described by the branch  
# pull all changes for the submodules
git submodule update --remote
 ---or---
# pull all changes in the repo including changes in the submodules
git pull --recurse-submodules


# update submodule in the master branch
# skip this if you use --recurse-submodules
# and have the master branch checked out
cd [submodule directory]
git checkout master
git pull

# commit the change in main repo
# to use the latest commit in master of the submodule
cd ..
git add [submodule directory]
git commit -m "move submodule to latest commit in master"

# share your changes
git push
``` 
