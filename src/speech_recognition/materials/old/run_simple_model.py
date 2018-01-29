from speech_recognition.dataset.data_iterators.audio_mfcc_google import *

from speech_recognition.dataset.preprocessor.simple_speech_preprocessor import *
from speech_recognition.dataset.preprocessor.speech_commands_scanner import *
from speech_recognition.models.conv.cnn_trad_fpool3_v1 import CNNTradFPoolConfigV1, CNNTradFPoolV1
from speech_recognition.models.conv.simple_conv import *

tf.logging.set_verbosity("INFO")

DATADIR = "../data/speech_commands_v0/"
# DATADIR = "/opt/dhiraa/sarvam/data/speech_commands_v0.01"
BATCH_SIZE = 8

# preprocessor = SimpleSpeechPreprocessor( data_dir=DATADIR,
#                  possible_speech_commands='yes no up down left right on off stop go silence unknown'.split(),
#                  batch_size=16)

sess = tf.InteractiveSession()


audio_sampling_settings = prepare_audio_sampling_settings(label_count=10,
                                                 sample_rate=SAMPLE_RATE,
                                                 clip_duration_ms=CLIP_DURATION_MS,
                                                 window_size_ms=WINDOW_SIZE_MS,
                                                 window_stride_ms=WINDOW_STRIDE_MS,
                                                 dct_coefficient_count=DCT_COEFFICIENT_COUNT)

audio_preprocessor = SpeechCommandsDirectoryProcessor(data_url="http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
                                                      data_dir=DATADIR,
                                                      silence_percentage=SILENCE_PERCENTAGE,
                                                      unknown_percentage=UNKNOWN_PERCENTAGE,
                                                      possible_commands=POSSIBLE_COMMANDS,
                                                      validation_percentage=VALIDATION_PERCENTAGE,
                                                      testing_percentage=TESTING_PERCENTAGE,
                                                      audio_sampling_settings=audio_sampling_settings)

# print_error(audio_preprocessor.get_train_data())
data_iterator = AudioMFCC(tf_sess=sess,
                          batch_size=BATCH_SIZE,
                          audio_preprocessor=audio_preprocessor,
                          audio_sampling_settings=audio_sampling_settings)

# model_config = SimpleSpeechRecognizerConfig()
model_config = CNNTradFPoolConfigV1(audio_sampling_settings)

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
# run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
run_config.allow_soft_placement = True
run_config.log_device_placement = False
run_config=tf.contrib.learn.RunConfig(session_config=run_config, model_dir=model_config._model_dir)

# model = SimpleSpeechRecognizer(model_config, run_config=run_config)
model = CNNTradFPoolV1(model_config, run_config=run_config)

def _create_my_experiment(run_config, hparams):
    exp = tf.contrib.learn.Experiment(
        estimator=model,
        train_input_fn=data_iterator.get_train_input_fn(),
        eval_input_fn=data_iterator.get_val_input_fn(),
        train_steps=10000, # just randomly selected params
        eval_steps=200,  # read source code for steps-epochs ariphmetics
        train_steps_per_iteration=1000,
    )
    return exp


tf.contrib.learn.learn_runner.run(
    experiment_fn=_create_my_experiment,
    run_config=run_config,
    schedule="continuous_train_and_eval",
    hparams=None)


# model.train(input_fn=data_iterator.get_train_input_fn(),
#                         hooks=[],
#                         max_steps=10000)

exit(-1)

# Testing

from tqdm import tqdm
# now we want to predict!
paths = glob(os.path.join(DATADIR, 'test/audio/*wav'))

def test_data_generator(data):
    def generator():
        for path in data:
            _, wav = wavfile.read(path)
            wav = wav.astype(np.float32) / np.iinfo(np.int16).max
            fname = os.path.basename(path)
            yield dict(
                sample=np.string_(fname),
                wav=wav,
            )

    return generator

test_input_fn = generator_input_fn(
    x=test_data_generator(paths),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_epochs=1,
    queue_capacity= 10 * BATCH_SIZE,
    num_threads=1,
)

# model = create_model(config=run_config, hparams=hparams)
it = model.predict(input_fn=test_input_fn)


# last batch will contain padding, so remove duplicates
submission = dict()
for t in tqdm(it):
    fname, label = t['sample'].decode(), audio_preprocessor.word_to_index[t['label']]
    submission[fname] = label

with open(os.path.join(model.model_dir, 'submission.csv'), 'w') as fout:
    fout.write('fname,label\n')
    for fname, label in submission.items():
        fout.write('{},{}\n'.format(fname, label))