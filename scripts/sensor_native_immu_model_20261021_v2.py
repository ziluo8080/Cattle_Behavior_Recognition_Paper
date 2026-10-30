"""Resume-capable version of the fixed single H-Inception fusion architecture."""
import tensorflow as tf
from aeon.networks import InceptionNetwork

from scripts.sensor_native_immu_model_20261021 import MaskedReadout, ParameterEMA, weighted_cross_entropy


@tf.keras.utils.register_keras_serializable(package='CowsNativeIMMU')
class CheckpointDropout(tf.keras.layers.Layer):
    """Stateless random masks with a counter saved as nontrainable model state."""
    def __init__(self, rate, seed, **kwargs):
        super().__init__(**kwargs)
        self.rate, self.seed = float(rate), int(seed)

    def build(self, input_shape):
        self.counter = self.add_weight(name='counter', shape=(), dtype=tf.int64,
                                       initializer='zeros', trainable=False)

    def call(self, inputs, training=False):
        if training:
            count = self.counter.assign_add(1)
            seed = tf.stack([tf.cast(self.seed, tf.int64), count])
            mask = tf.random.stateless_uniform(tf.shape(inputs), seed, dtype=inputs.dtype) >= self.rate
            return tf.where(mask, inputs / (1 - self.rate), 0.)
        return inputs

    def get_config(self):
        return dict(super().get_config(), rate=self.rate, seed=self.seed)


def make_model(kind='fusion', max_age=.2, seed=17):
    if kind not in ('fusion', 'dynamic_only', 'context_only', 'late_logits'):
        raise ValueError(kind)
    dynamic = tf.keras.Input((100, 8), name='dynamic')
    context = tf.keras.Input((136,), name='context')
    fast = slow = None
    if kind != 'context_only':
        valid = dynamic[:, :, 6] > .5
        cleaned = tf.concat([tf.where(valid[:, :, None], dynamic[:, :, :6], 0.),
                             tf.cast(valid[:, :, None], tf.float32),
                             tf.where(valid[:, :, None], tf.clip_by_value(dynamic[:, :, 7:], 0., max_age), max_age)], axis=2)
        network = InceptionNetwork(n_filters=32, kernel_size=40, depth=6,
                                   use_residual=True, use_custom_filters=True)
        encoder_input, pooled = network.build_network((100, 8))
        gap = pooled._keras_history.layer
        if not isinstance(gap, tf.keras.layers.GlobalAveragePooling1D):
            raise RuntimeError('Unexpected pinned upstream encoder output')
        encoder = tf.keras.Model(encoder_input, gap.input, name='single_h_inception_encoder')
        assert encoder.output_shape == (None, 100, 128)
        fast = MaskedReadout(name='dynamic_readout')([encoder(cleaned), valid])
    if kind != 'dynamic_only':
        slow = tf.keras.layers.Dense(64, activation='relu', name='context_embedding')(context)

    def head(value, prefix, index):
        hidden = tf.keras.layers.Dense(128, activation='relu', name=prefix + '_hidden')(value)
        hidden = CheckpointDropout(.1, seed + index, name=prefix + '_dropout')(hidden)
        return tf.keras.layers.Dense(7, name=prefix + '_logits')(hidden)

    if kind == 'fusion':
        logits = head(tf.concat([fast, slow], axis=1), 'joint', 1)
    elif kind == 'late_logits':
        logits = .5 * (head(fast, 'dynamic_head', 1) + head(slow, 'context_head', 2))
    else:
        logits = head(fast if kind == 'dynamic_only' else slow, 'single', 1)
    return tf.keras.Model(dict(dynamic=dynamic, context=context), logits, name=kind)
