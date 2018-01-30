# Introduction
Speech recognition is one of nicehe field which allows humans to interact with
machines in a sophisticated lazy way. 
Here we are trying to understand, experiment and try to create user friendly apps
 on those fronts with Tensorflow models.

# Related Work
- https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html
- https://www.tensorflow.org/versions/master/tutorials/audio_recognition
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands/

## Android App
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android/
- ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/tensorflow_demo.apk
- https://github.com/petewarden/open-speech-recording

# Problem Statement

To come up with an simple easy to use software environment to train on audio data with plug and play
 modules for data pre-processing, training different models and serving the pre-trained models on 
 web and mobile devices.
 
# Proposed Solution
Come up with following modular components which can be then used as plug and play components:
 - Dataset Modules with preprocessing Modules
 - Data Iterator Modules
 - Tensorflow Models
 - Tensorflow Model serving
    - Web app
    - Mobile
    
# Validation


-----------------------------------------------------------------------------------------------------------
# Learning

Do checkout [here](https://dhiraa.github.io/sarvam//deep_learning/audio/basics/) 
for basic understanding of Audio Dataset to work with Deep Learning!

# Dataset

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

```
my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav
```

Sample data set used here is from [http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)

# Libraries
- https://github.com/librosa/librosa
- https://github.com/CPJKU/madmom

# [Models](models)
- [Convolution Net Based Models](models/conv/)

```

#sample how to run. For more check on models!
cd /path/to/sarvam/src/

python speech_recognition/commands/run_experiments.py \
--mode=train \
--dataset-name=speech_commands_v0 \
--data-iterator-name=audio_mfcc_google \
--model-name=cnn_trad_fpool3 \
--batch-size=32 \
--num-epochs=5

tensorboard logdir=experiments/CNNTradFPool/

```