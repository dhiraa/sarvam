import numpy as np
import tensorflow as tf
import math
from PIL import Image
from tensorflow.contrib import layers
from tensorflow.contrib import signal
from tqdm import tqdm

from asariri.dataset.features.asariri_features import GANFeature
from asariri.asariri_utils.asariri_config import *
from sarvam.helpers.print_helper import *
from speech_recognition.sr_config.sr_config import *
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.training import session_run_hook
import collections
from tensorflow.python.training import training_util
from matplotlib import pyplot


"""
Paper: 
    - https://arxiv.org/abs/1703.10717
Git: 
    - https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/BEGAN.py
    - https://github.com/carpedm20/BEGAN-tensorflow
"""

class BEGANConfig(ModelConfigBase):
    def __init__(self, model_dir, batch_size, num_image_channels):
        self._model_dir = model_dir

        self.num_image_channels = num_image_channels

        self.learning_rate = 0.001
        self.alpha = 0.15
        self.beta1 = 0.4
        self.z_dim = 30

    @staticmethod
    def user_config(batch_size, data_iterator):
        _model_dir = EXPERIMENT_MODEL_ROOT_DIR + "/" + data_iterator.name + "/VanillaGAN/"
        config = BEGANConfig(_model_dir, batch_size, data_iterator._dataset.num_channels)
        BEGANConfig.dump(_model_dir, config)
        return config
