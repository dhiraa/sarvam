from tqdm import tqdm
from sarvam.helpers.print_helper import *
import numpy as np
from scipy.io import wavfile
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

from asariri.dataset.features.asariri_features import AudioImageFeature


class RawDataIterator:
    def __init__(self, batch_size, num_epochs, preprocessor):
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._preprocessor = preprocessor
        self._feature_type = AudioImageFeature


    def data_generator(self, data, params, mode='train'):
        def generator():
            if mode == 'train':
                np.random.shuffle(data)
            # Feel free to add any augmentation
            for i, data_dict in tqdm(enumerate(data), desc=mode):
                audio_file_name = data_dict["audio"]
                image_file_name = data_dict["image"]
                person_name = data_dict["label"]

                try:
                    sample_rate, wav = wavfile.read(audio_file_name)
                    wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                    L = 16000 * 5 # be aware, some files are shorter than 1 sec!
                    if len(wav) < L:
                        continue

                    # print_error(i)
                    # print(fname)
                    # print_info(wav)
                    # print_debug(wav.sum())

                    yield {self._feature_type.FEATURE_AUDIO : wav,
                     self._feature_type.FEATURE_IMAGE: }

                except Exception as err:
                    print_error(str(err) + " " + str(person_name) + " " + audio_file_name + " " + image_file_name)

        return generator

    def get_train_input_fn(self):
        train_input_fn = generator_input_fn(
            x=self.data_generator(self._preprocessor.get_train_files(), None, 'train'),
            target_key=self._feature_type.FEATURE_IMAGE,  # you could leave target_key in features, so labels in model_handler will be empty
            batch_size=self._batch_size,
            shuffle=True,
            num_epochs=self._num_epochs,
            queue_capacity=3 * self._batch_size + 10,
            num_threads=1,
        )

        return train_input_fn

    def get_val_input_fn(self):
        val_input_fn = generator_input_fn(
            x=self.data_generator(self._preprocessor.get_val_files(), None, 'val'),
            target_key=self._feature_type.FEATURE_IMAGE,
            batch_size=self._batch_size,
            shuffle=True,
            num_epochs=1,
            queue_capacity=3 * self._batch_size + 10,
            num_threads=1,
        )

        return val_input_fn
