import os
import tensorflow as tf
import math
import re
import hashlib
import random
import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

from speech_recognition.dataset.downloader import maybe_download_and_extract_dataset
from speech_recognition.sr_config.sr_config import *

from sarvam.helpers.print_helper import *

def prepare_words_list(wanted_words):
    """Prepends common tokens to the custom word list.
    Args:
      wanted_words: List of strings containing the custom words.
    Returns:
      List with the standard silence and unknown tokens added.
    """
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words

def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.
    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.
    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.
    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.
    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result

class AudioProcessor(object):
    """Handles loading, partitioning, and preparing audio training data."""

    def __init__(self, data_url,
                 data_dir,
                 silence_percentage,
                 unknown_percentage,
                 wanted_words,
                 validation_percentage,
                 testing_percentage,
                 model_settings):

        self.data_dir = data_dir

        print_error(wanted_words)
        wanted_words = wanted_words.split(",")
        # wanted_words = prepare_words_list(wanted_words)
        print_error(wanted_words)

        random.seed(RANDOM_SEED)
        desired_samples = model_settings['desired_samples']

        self.data_index = {'validation': [], 'testing': [], 'training': []}

        self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
        self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
        self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
        self.background_data_placeholder_ = tf.placeholder(tf.float32, [desired_samples, 1])
        self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])

        maybe_download_and_extract_dataset(data_url, data_dir)

        self.prepare_data_index(silence_percentage,
                                unknown_percentage,
                                wanted_words,
                                validation_percentage,
                                testing_percentage)
        self.prepare_background_data()
        self.prepare_processing_graph(model_settings)



    def prepare_data_index(self,
                           silence_percentage,
                           unknown_percentage,
                           wanted_words,
                           validation_percentage,
                           testing_percentage):
        """
        Prepares a list of the samples organized by set and label.
        The training loop needs a list of all the available data, organized by
        which partition it should belong to, and with ground truth labels attached.
        This function analyzes the folders below the `data_dir`, figures out the
        right labels for each file based on the name of the subdirectory it belongs to,
        and uses a stable hash to assign it to a data set partition.
        :param silence_percentage: How much of the resulting data should be background.
        :param unknown_percentage: How much should be audio outside the wanted classes.
        :param wanted_words: Labels of the classes we want to be able to recognize.
        :param validation_percentage: How much of the data set to use for validation.
        :param testing_percentage: How much of the data set to use for testing.
        :return:  Dictionary containing a list of file information for each set partition,
          and a lookup map for each class to determine its numeric index.
        :raises Exception: If expected files are not found.
        """

        # Make sure the shuffling and picking of unknowns is deterministic.

        wanted_words_index = {}

        for index, wanted_word in enumerate(wanted_words):
            wanted_words_index[wanted_word] = index + 2


        unknown_index = {'validation': [], 'testing': [], 'training': []}
        all_words = {}

        # Look through all the subfolders to find audio samples
        search_path = os.path.join(self.data_dir, '*', '*.wav')

        for wav_path in gfile.Glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            # Treat the '_background_noise_' folder as a special case, since we expect
            # it to contain long audio samples we mix in to improve training.
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = which_set(wav_path, validation_percentage, testing_percentage)
            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            if word in wanted_words_index:
                self.data_index[set_index].append({'label': word, 'file': wav_path})
            else:
                unknown_index[set_index].append({'label': word, 'file': wav_path})

        if not all_words:
            raise Exception('No .wavs found at ' + search_path)
        print_error(all_words)
        for index, wanted_word in enumerate(wanted_words):
            if wanted_word not in all_words:
                raise Exception('Expected to find ' + wanted_word +
                                ' in labels but only found ' +
                                ', '.join(all_words.keys()))

        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index['training'][0]['file']

        for set_index in ['validation', 'testing', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append({
                    'label': SILENCE_LABEL,
                    'file': silence_wav_path
                })
            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])

        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        self.words_list = prepare_words_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def prepare_background_data(self):
        """
        Searches a folder for background noise audio, and loads it into memory.
        It's expected that the background audio samples will be in a subdirectory
        named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
        the sample rate of the training data, but can be much longer in duration.
        If the '_background_noise_' folder doesn't exist at all, this isn't an
        error, it's just taken to mean that no background noise augmentation should
        be used. If the folder does exist, but it's empty, that's treated as an
        error.
        :return: List of raw PCM-encoded audio samples of background noise.
        :raises: Exception: If files aren't found in the folder.
        """

        self.background_data = []

        background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        if not os.path.exists(background_dir):
            return self.background_data

        with tf.Session(graph=tf.Graph()) as sess:
            wav_filename_placeholder = tf.placeholder(tf.string, [])
            wav_loader = io_ops.read_file(wav_filename_placeholder)
            wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
            search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,
                                       '*.wav')
            for wav_path in gfile.Glob(search_path):
                wav_data = sess.run(
                    wav_decoder,
                    feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
                self.background_data.append(wav_data)
            if not self.background_data:
                raise Exception('No background wav files were found in ' + search_path)

    def prepare_processing_graph(self, model_settings):
        """
        Builds a TensorFlow graph to apply the input distortions.
        Creates a graph that loads a WAVE file, decodes it, scales the volume,
        shifts it in time, adds in background noise, calculates a spectrogram, and
        then builds an MFCC fingerprint from that.
        This must be called with an active TensorFlow session running, and it
        creates multiple placeholder inputs, and one output:
          - wav_filename_placeholder_: Filename of the WAV to load.
          - foreground_volume_placeholder_: How loud the main clip should be.
          - time_shift_padding_placeholder_: Where to pad the clip.
          - time_shift_offset_placeholder_: How much to move the clip in time.
          - background_data_placeholder_: PCM sample data for background noise.
          - background_volume_placeholder_: Loudness of mixed-in background.
          - mfcc_: Output 2D fingerprint of processed audio.
        :param model_settings: Information about the current model being trained.
        :return: 
        """

        desired_samples = model_settings['desired_samples']

        wav_loader = io_ops.read_file(self.wav_filename_placeholder_)

        wav_decoder = contrib_audio.decode_wav(
            wav_loader, desired_channels=1, desired_samples=desired_samples)
        # Allow the audio sample's volume to be adjusted.
        self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
        scaled_foreground = tf.multiply(wav_decoder.audio,
                                        self.foreground_volume_placeholder_)
        # Shift the sample's start position, and pad any gaps with zeros.

        padded_foreground = tf.pad(
            scaled_foreground,
            self.time_shift_padding_placeholder_,
            mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground,
                                     self.time_shift_offset_placeholder_,
                                     [desired_samples, -1])
        # Mix in background noise.

        background_mul = tf.multiply(self.background_data_placeholder_,
                                     self.background_volume_placeholder_)
        background_add = tf.add(background_mul, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
        # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
        spectrogram = contrib_audio.audio_spectrogram(
            background_clamp,
            window_size=model_settings['window_size_samples'],
            stride=model_settings['window_stride_samples'],
            magnitude_squared=True)
        self.mfcc_ = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=model_settings['dct_coefficient_count'])

    def set_size(self, mode):
        """Calculates the number of samples in the dataset partition.
        Args:
          mode: Which partition, must be 'training', 'validation', or 'testing'.
        Returns:
          Number of samples in the partition.
        """
        return len(self.data_index[mode])

    def get_data(self,
                 how_many,
                 offset,
                 model_settings,
                 background_frequency,
                 background_volume_range,
                 time_shift,
                 mode,
                 sess):
        """
        Gather samples from the data set, applying transformations as needed.
        When the mode is 'training', a random selection of samples will be returned,
        otherwise the first N clips in the partition will be used. This ensures that
        validation always uses the same samples, reducing noise in the metrics.
        :param how_many: Desired number of samples to return. -1 means the entire
            contents of this partition. 
        :param offset: Where to start when fetching deterministically.
          model_settings: Information about the current model being trained.
        :param model_settings: 
        :param background_frequency: How many clips will have background noise, 0.0 to
            1.0.
        :param background_volume_range: How loud the background noise will be.
        :param time_shift: How much to randomly shift the clips by in time.
        :param mode: Which partition to use, must be 'training', 'validation', or
            'testing'.
        :param sess: TensorFlow session that was active when processor was created.
        :return: List of sample data for the transformed samples, and list of label indexes
        """
        # Pick one of the partitions to choose samples from.

        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))

        # Data and labels will be populated and returned.
        data = np.zeros((sample_count, model_settings['fingerprint_size']))
        labels = np.zeros(sample_count)
        desired_samples = model_settings['desired_samples']
        use_background = self.background_data and (mode == 'training')
        pick_deterministically = (mode != 'training')

        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in xrange(offset, offset + sample_count):
            # Pick which audio sample to use.
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]
            # If we're time shifting, set up the offset for this sample.
            if time_shift > 0:
                time_shift_amount = np.random.randint(-time_shift, time_shift)
            else:
                time_shift_amount = 0
            if time_shift_amount > 0:
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]
            input_dict = {
                self.wav_filename_placeholder_: sample['file'],
                self.time_shift_padding_placeholder_: time_shift_padding,
                self.time_shift_offset_placeholder_: time_shift_offset,
            }
            # Choose a section of background noise to mix in.
            if use_background:
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                background_offset = np.random.randint(
                    0, len(background_samples) - model_settings['desired_samples'])
                background_clipped = background_samples[background_offset:(
                    background_offset + desired_samples)]
                background_reshaped = background_clipped.reshape([desired_samples, 1])
                if np.random.uniform(0, 1) < background_frequency:
                    background_volume = np.random.uniform(0, background_volume_range)
                else:
                    background_volume = 0
            else:
                background_reshaped = np.zeros([desired_samples, 1])
                background_volume = 0
            input_dict[self.background_data_placeholder_] = background_reshaped
            input_dict[self.background_volume_placeholder_] = background_volume
            # If we want silence, mute out the main sample but leave the background.
            if sample['label'] == SILENCE_LABEL:
                input_dict[self.foreground_volume_placeholder_] = 0
            else:
                input_dict[self.foreground_volume_placeholder_] = 1
            # Run the graph to produce the output audio.
            data[i - offset, :] = sess.run(self.mfcc_, feed_dict=input_dict).flatten()
            label_index = self.word_to_index[sample['label']]
            labels[i - offset] = label_index
        return data, labels

    def get_unprocessed_data(self, how_many, model_settings, mode):
        """Retrieve sample data for the given partition, with no transformations.
        Args:
          how_many: Desired number of samples to return. -1 means the entire
            contents of this partition.
          model_settings: Information about the current model being trained.
          mode: Which partition to use, must be 'training', 'validation', or
            'testing'.
        Returns:
          List of sample data for the samples, and list of labels in one-hot form.
        """
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = how_many
        desired_samples = model_settings['desired_samples']

        words_list = self.words_list

        data = np.zeros((sample_count, desired_samples))
        labels = []

        with tf.Session(graph=tf.Graph()) as sess:
            wav_filename_placeholder = tf.placeholder(tf.string, [])
            wav_loader = io_ops.read_file(wav_filename_placeholder)
            wav_decoder = contrib_audio.decode_wav(
                wav_loader, desired_channels=1, desired_samples=desired_samples)
            foreground_volume_placeholder = tf.placeholder(tf.float32, [])
            scaled_foreground = tf.multiply(wav_decoder.audio,
                                            foreground_volume_placeholder)
            for i in range(sample_count):
                if how_many == -1:
                    sample_index = i
                else:
                    sample_index = np.random.randint(len(candidates))
                sample = candidates[sample_index]
                input_dict = {wav_filename_placeholder: sample['file']}
                if sample['label'] == SILENCE_LABEL:
                    input_dict[foreground_volume_placeholder] = 0
                else:
                    input_dict[foreground_volume_placeholder] = 1
                data[i, :] = sess.run(scaled_foreground, feed_dict=input_dict).flatten()
                label_index = self.word_to_index[sample['label']]
                labels.append(words_list[label_index])
        return data, labels