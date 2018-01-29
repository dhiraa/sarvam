import tensorflow as tf
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import signal
# it's a magic function :)
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
from tqdm import tqdm

def baseline(x, params, is_training):
    x = layers.batch_norm(x, is_training=is_training)
    for i in range(4):
        x = layers.conv2d(
            x, 16 * (2 ** i), 3, 1,
            activation_fn=tf.nn.elu,
            normalizer_fn=layers.batch_norm if params.use_batch_norm else None,
            normalizer_params={'is_training': is_training}
        )
        x = layers.max_pool2d(x, 2, 2)

    # just take two kind of pooling and then mix them, why not :)
    mpool = tf.reduce_max(x, axis=[1, 2], keep_dims=True)
    apool = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

    x = 0.5 * (mpool + apool)
    # we can use conv2d 1x1 instead of dense
    x = layers.conv2d(x, 128, 1, 1, activation_fn=tf.nn.elu)
    x = tf.nn.dropout(x, keep_prob=params.keep_prob if is_training else 1.0)

    # again conv2d 1x1 instead of dense layer
    logits = layers.conv2d(x, params.num_classes, 1, 1, activation_fn=None)
    return tf.squeeze(logits, [1, 2])


# features is a dict with keys: tensors from our datagenerator
# labels also were in features, but excluded in generator_input_fn by target_key

def model_handler(features, labels, mode, params, config):
    # Im really like to use make_template instead of variable_scopes and re-usage
    extractor = tf.make_template(
        'extractor', baseline,
        create_scope_now_=True,
    )
    # wav is a waveform signal with shape (16000, )
    wav = features['wav']
    # we want to compute spectograms by means of short time fourier transform:
    specgram = signal.stft(
        wav,
        400,  # 16000 [samples per second] * 0.025 [s] -- default stft window frame
        160,  # 16000 * 0.010 -- default stride
    )
    # specgram is a complex tensor, so split it into abs and phase parts:
    phase = tf.angle(specgram) / np.pi
    # log(1 + abs) is a default transformation for energy units
    amp = tf.log1p(tf.abs(specgram))

    x = tf.stack([amp, phase], axis=3)  # shape is [bs, time, freq_bins, 2]
    x = tf.to_float(x)  # we want to have float32, not float64

    logits = extractor(x, params, mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        # some lr tuner, you could use move interesting functions
        def learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate, global_step, decay_steps=10000, decay_rate=0.99)

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
            learning_rate_decay_fn=learning_rate_decay_fn,
            clip_gradients=params.clip_gradients,
            variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        specs = dict(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        prediction = tf.argmax(logits, axis=-1)
        acc, acc_op = tf.metrics.mean_per_class_accuracy(
            labels, prediction, params.num_classes)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        specs = dict(
            mode=mode,
            loss=loss,
            eval_metric_ops=dict(
                acc=(acc, acc_op),
            )
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'label': tf.argmax(logits, axis=-1),  # for probability just take tf.nn.softmax()
            'sample': features['sample'],  # it's a hack for simplicity
        }
        specs = dict(
            mode=mode,
            predictions=predictions,
        )
    return tf.estimator.EstimatorSpec(**specs)


def create_model(config=None, hparams=None):
    return tf.estimator.Estimator(
        model_fn=model_handler,
        config=config,
        params=hparams,
    )