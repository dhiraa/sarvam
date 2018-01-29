import os
import re
import numpy as np
from scipy.io import wavfile
from glob import glob
from tqdm import tqdm
# it's a magic function :)
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

class SimpleSpeechPreprocessor():
    def __init__(self,
                 data_dir,
                 possible_speech_commands,
                 batch_size):

        self.BATCH_SIZE = batch_size
        self.DATA_DIR = data_dir
        self.POSSIBLE_SPEECH_COMMANDS = possible_speech_commands

        self.ID2NAME = {i: name for i, name in enumerate(self.POSSIBLE_SPEECH_COMMANDS)}
        self.NAME2ID = {name: i for i, name in self.ID2NAME.items()}
        self.word_to_index = {name: i for i, name in self.ID2NAME.items()}


        self.TRAIN_SET, self.VALSET = self.load_data()

    def load_data(self):
        """ Return 2 lists of tuples:
        [(class_id, user_id, path), ...] for train
        [(class_id, user_id, path), ...] for validation
        """
        # Just a simple regexp for paths with three groups:
        # prefix, label, user_id

        pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
        all_files = glob(os.path.join(self.DATA_DIR, 'train/audio/*/*wav'))

        with open(os.path.join(self.DATA_DIR, 'train/validation_list.txt'), 'r') as fin:
            validation_files = fin.readlines()

        valset = set()

        for entry in tqdm(validation_files):
            r = re.match(pattern, entry)
            if r:
                valset.add(r.group(3))

        possible = set(self.POSSIBLE_SPEECH_COMMANDS)

        train, val = [], []
        for entry in tqdm(all_files):
            r = re.match(pattern, entry)
            if r:
                label, uid = r.group(2), r.group(3)
                if label == '_background_noise_':
                    label = '_silence_'
                if label not in possible:
                    label = 'unknown'

                label_id = self.NAME2ID[label]

                # sample = (label_id, uid, entry)
                sample = {"label" : label, "file" : entry}

                if uid in valset:
                    val.append(sample)
                else:
                    train.append(sample)

        print('There are {} train and {} val samples'.format(len(train), len(val)))
        return train, val


    def get_train_files(self):
        return self.TRAIN_SET

    def get_val_files(self):
        return self.VALSET

    def get_test_files(self):
        return self.data_buckets["testing"]

    def data_generator(self, data, params, mode='train'):
        def generator():
            if mode == 'train':
                np.random.shuffle(data)
            # Feel free to add any augmentation
            for (label_id, uid, fname) in tqdm(data):
                try:
                    _, wav = wavfile.read(fname)
                    wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                    L = 16000  # be aware, some files are shorter than 1 sec!
                    if len(wav) < L:
                        continue
                    # let's generate more silence!
                    samples_per_file = 1 if label_id != self.NAME2ID['silence'] else 20
                    for _ in range(samples_per_file):
                        if len(wav) > L:
                            beg = np.random.randint(0, len(wav) - L)
                        else:
                            beg = 0
                        yield dict(
                            target=np.int32(label_id),
                            wav=wav[beg: beg + L],
                        )
                except Exception as err:
                    print(err, label_id, uid, fname)

        return generator


    def get_train_input_fn(self):
        train_input_fn = generator_input_fn(
            x=self.data_generator(self.TRAIN_SET, None, 'train'),
            target_key='target',  # you could leave target_key in features, so labels in model_handler will be empty
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_epochs=None,
            queue_capacity=3 * self.BATCH_SIZE + 10,
            num_threads=1,
        )

        return train_input_fn

    def get_val_input_fn(self):
        val_input_fn = generator_input_fn(
            x=self.data_generator(self.VALSET, None, 'val'),
            target_key='target',
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_epochs=None,
            queue_capacity=3 * self.BATCH_SIZE + 10,
            num_threads=1,
        )

        return val_input_fn
