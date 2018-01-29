from sarvam.config.config_helper import ConfigManager

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0

UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1

BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185


sr_user_config = ConfigManager("speech_recognition/sr_config/config.ini")

SILENCE_PERCENTAGE      = sr_user_config.get_item_as_float("default", "silence_percentage")
UNKNOWN_PERCENTAGE      = sr_user_config.get_item_as_float("default", "unknown_percentage")
TESTING_PERCENTAGE      = sr_user_config.get_item_as_float("default", "testing_percentage")
VALIDATION_PERCENTAGE   = sr_user_config.get_item_as_float("default", "validation_percentage")

BACKGROUND_VOLUME       = sr_user_config.get_item_as_float("default", "background_volume")
BACKGROUND_FREQUENCY    = sr_user_config.get_item_as_float("default", "background_frequency")
TIME_SHIFT_MS           = sr_user_config.get_item_as_float("default", "time_shift_ms")
CLIP_DURATION_MS        = sr_user_config.get_item_as_float("default", "clip_duration_ms")
WINDOW_SIZE_MS          = sr_user_config.get_item_as_float("default", "window_size_ms")
WINDOW_STRIDE_MS        = sr_user_config.get_item_as_float("default", "window_stride_ms")

SAMPLE_RATE             = sr_user_config.get_item_as_int("default", "sample_rate")
DCT_COEFFICIENT_COUNT   = sr_user_config.get_item_as_int("default", "dct_coefficient_count")
HOW_MANY_TRAINING_STEPS = sr_user_config.get_item_as_int("default", "how_many_training_steps")
EVAL_STEP_INTERVAL      = sr_user_config.get_item_as_int("default", "eval_step_interval")

POSSIBLE_COMMANDS            = sr_user_config.get_item("default", "possible_commands").split(",")

TIME_SHIFT_SAMPLES      = int((TIME_SHIFT_MS * SAMPLE_RATE) / 1000)


def prepare_words_list(possible_speech_commands):
    """Prepends common tokens to the custom word list.
    Args:
      possible_speech_commands: List of strings containing the custom words.
    Returns:
      List with the standard silence and unknown tokens added.
    """
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + possible_speech_commands




def prepare_audio_sampling_settings(label_count,
                                    sample_rate,
                                    clip_duration_ms,
                                    window_size_ms,
                                    window_stride_ms,
                                    dct_coefficient_count):
  """Calculates common settings needed for all models.
  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second. (Default: 16000)
    clip_duration_ms: Length of each audio clip to be analyzed. (Default: 1000)
    window_size_ms: Duration of frequency analysis window. (Default: 30)
    window_stride_ms: How far to move in time between frequency windows. (Default: 10)
    dct_coefficient_count: Number of frequency bins to use for analysis. (Default: 40)
  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000) # 16000
  window_size_samples = int(sample_rate * window_size_ms / 1000) # 16000 * 30 / 1000 = 180
  window_stride_samples = int(sample_rate * window_stride_ms / 1000) # 16000 * 10 / 1000 = 160
  length_minus_window = (desired_samples - window_size_samples) # 16000 - 180 = 15020

  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples) # 1 + (15020/160) = 94

  fingerprint_size = dct_coefficient_count * spectrogram_length # 40 * 94 = 3760

  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }
