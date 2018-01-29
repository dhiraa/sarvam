from speech_recognition.dataset.data_iterators.audio_mfcc import *

from speech_recognition.dataset.preprocessor.speech_commands_scanner import *
from speech_recognition.materials.old.google_speech_processor import AudioProcessor

sess = tf.InteractiveSession()

audio_sampling_settings = prepare_audio_sampling_settings(label_count=10,
                                                 sample_rate=SAMPLE_RATE,
                                                 clip_duration_ms=CLIP_DURATION_MS,
                                                 window_size_ms=WINDOW_SIZE_MS,
                                                 window_stride_ms=WINDOW_STRIDE_MS,
                                                 dct_coefficient_count=DCT_COEFFICIENT_COUNT)

audio_preprocessor = AudioProcessor(data_url="http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
                                    data_dir="../data/tensorflow_speech_recoginition_challenge/train/audio/",
                                    silence_percentage=SILENCE_PERCENTAGE,
                                    unknown_percentage=UNKNOWN_PERCENTAGE,
                                    wanted_words=POSSIBLE_COMMANDS,
                                    validation_percentage=VALIDATION_PERCENTAGE,
                                    testing_percentage=TESTING_PERCENTAGE,
                                    audio_sampling_settings=audio_sampling_settings)

res = audio_preprocessor.get_data(how_many=16,
                                  offset=0,
                                  audio_sampling_settings=audio_sampling_settings,
                                  background_frequency=BACKGROUND_FREQUENCY,
                                  background_volume_range=BACKGROUND_VOLUME,
                                  time_shift=TIME_SHIFT_MS,
                                  mode="training",
                                  sess=sess)

print(res[0])
print(res[1])

print(res[0].shape)
print(res[1].shape)

# audio_preprocessor = SpeechCommandsDirectoryProcessor(data_url="http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
#                                     data_dir="../data/tensorflow_speech_recoginition_challenge/train/audio/",
#                                     silence_percentage=SILENCE_PERCENTAGE,
#                                     unknown_percentage=UNKNOWN_PERCENTAGE,
#                                     wanted_words=POSSIBLE_COMMANDS,
#                                     validation_percentage=VALIDATION_PERCENTAGE,
#                                     testing_percentage=TESTING_PERCENTAGE,
#                                     audio_sampling_settings=audio_sampling_settings)
#
# data_iterator = AudioMFCC(batch_size=16, audio_preprocessor=audio_preprocessor)
#
# data_iterator.get_train_input_function()