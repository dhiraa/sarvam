import sys
sys.path.append("../")
from conv_v0 import *
from audio_utils.dataset import *

params=dict(
    seed=2018,
    batch_size=64,
    keep_prob=0.5,
    learning_rate=1e-3,
    clip_gradients=15.0,
    use_batch_norm=True,
    num_classes=len(POSSIBLE_LABELS),
)

hparams = tf.contrib.training.HParams(**params)
os.makedirs(os.path.join(OUT_DIR, 'eval'), exist_ok=True)
model_dir = OUT_DIR

run_config = tf.contrib.learn.RunConfig(model_dir=model_dir)

trainset, valset = load_data(DATA_DIR)

train_input_fn = generator_input_fn(
    x=data_generator(trainset, hparams, 'train'),
    target_key='target',  # you could leave target_key in features, so labels in model_handler will be empty
    batch_size=hparams.batch_size, shuffle=True, num_epochs=None,
    queue_capacity=3 * hparams.batch_size + 10, num_threads=1,
)

val_input_fn = generator_input_fn(
    x=data_generator(valset, hparams, 'val'),
    target_key='target',
    batch_size=hparams.batch_size, shuffle=True, num_epochs=None,
    queue_capacity=3 * hparams.batch_size + 10, num_threads=1,
)


def _create_my_experiment(run_config, hparams):
    exp = tf.contrib.learn.Experiment(
        estimator=create_model(config=run_config, hparams=hparams),
        train_input_fn=train_input_fn,
        eval_input_fn=val_input_fn,
        train_steps=10000,  # just randomly selected params
        eval_steps=200,  # read source code for steps-epochs ariphmetics
        train_steps_per_iteration=1000,
    )
    return exp


tf.contrib.learn.learn_runner.run(
    experiment_fn=_create_my_experiment,
    run_config=run_config,
    schedule="continuous_train_and_eval",
    hparams=hparams)



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
    batch_size=hparams.batch_size,
    shuffle=False,
    num_epochs=1,
    queue_capacity= 10 * hparams.batch_size,
    num_threads=1,
)

model = create_model(config=run_config, hparams=hparams)
it = model.predict(input_fn=test_input_fn)


# last batch will contain padding, so remove duplicates
submission = dict()
for t in tqdm(it):
    fname, label = t['sample'].decode(), id2name[t['label']]
    submission[fname] = label

with open(os.path.join(model_dir, 'submission.csv'), 'w') as fout:
    fout.write('fname,label\n')
    for fname, label in submission.items():
        fout.write('{},{}\n'.format(fname, label))