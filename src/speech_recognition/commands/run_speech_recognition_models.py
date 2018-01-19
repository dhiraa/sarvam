import tensorflow as tf
from speech_recognition.dataset.preprocessor.audio_processor import AudioProcessor
from speech_recognition.sr_config.sr_config import *

sess = tf.InteractiveSession()

model_settings = prepare_model_settings(label_count=10,
                           sample_rate=SAMPLE_RATE,
                           clip_duration_ms=CLIP_DURATION_MS,
                           window_size_ms=WINDOW_SIZE_MS,
                           window_stride_ms=WINDOW_STRIDE_MS,
                           dct_coefficient_count=DCT_COEFFICIENT_COUNT)

audio_preprocessor = AudioProcessor( data_url="http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
                 data_dir="../data/tensorflow_speech_recoginition_challenge/train/audio/",
                 silence_percentage=SILENCE_PERCENTAGE,
                 unknown_percentage=UNKNOWN_PERCENTAGE,
                 wanted_words=WANTED_WORDS,
                 validation_percentage=VALIDATION_PERCENTAGE,
                 testing_percentage=TESTING_PERCENTAGE,
                 model_settings=model_settings)

print(audio_preprocessor.get_data(how_many=16,
                 offset=0,
                 model_settings=model_settings,
                 background_frequency=BACKGROUND_FREQUENCY,
                 background_volume_range=BACKGROUND_VOLUME,
                 time_shift=TIME_SHIFT_MS,
                 mode="training",
                 sess=sess))