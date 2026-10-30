"""Single aeon H-Inception encoder with masked readouts and feature fusion."""
import math
from contextlib import contextmanager

import tensorflow as tf
from aeon.networks import InceptionNetwork


@tf.keras.utils.register_keras_serializable(package='CowsNativeIMMU')
class MaskedReadout(tf.keras.layers.Layer):
    def call(self, inputs):
        features, valid = inputs
        outputs = []
        for start in (0, 90):
            mask = tf.cast(valid[:, start:, None], tf.bool)
            total = tf.reduce_sum(tf.where(mask, features[:, start:], 0.), axis=1)
            count = tf.reduce_sum(tf.cast(mask, features.dtype), axis=1)
            outputs.append(tf.math.divide_no_nan(total, count))
        return tf.concat(outputs, axis=1)


def make_model(kind='fusion', max_age=.2):
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
            raise RuntimeError('Unexpected upstream network output; inspect pinned source')
        feature_encoder = tf.keras.Model(encoder_input, gap.input, name='single_h_inception_encoder')
        assert feature_encoder.output_shape == (None, 100, 128)
        features = feature_encoder(cleaned)
        fast = MaskedReadout(name='dynamic_readout')([features, valid])
    if kind != 'dynamic_only':
        slow = tf.keras.layers.Dense(64, activation='relu', name='context_embedding')(context)

    def head(value, prefix):
        hidden = tf.keras.layers.Dense(128, activation='relu', name=prefix + '_hidden')(value)
        hidden = tf.keras.layers.Dropout(.1, name=prefix + '_dropout')(hidden)
        return tf.keras.layers.Dense(7, name=prefix + '_logits')(hidden)

    if kind == 'fusion':
        logits = head(tf.concat([fast, slow], axis=1), 'joint')
    elif kind == 'late_logits':
        logits = .5 * (head(fast, 'dynamic_head') + head(slow, 'context_head'))
    else:
        logits = head(fast if kind == 'dynamic_only' else slow, 'single')
    return tf.keras.Model(dict(dynamic=dynamic, context=context), logits, name=kind)


def weighted_cross_entropy(labels, logits, class_weights):
    labels = tf.cast(labels, tf.int32)
    weights = tf.gather(tf.cast(class_weights, logits.dtype), labels)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.math.divide_no_nan(tf.reduce_sum(weights * losses), tf.reduce_sum(weights))


class ParameterEMA:
    """Average trainable variables; copy fixed filters and BN moving statistics."""
    def __init__(self, model, decay=.998, warmup_steps=200):
        self.model = model
        self.decay, self.warmup_steps = decay, warmup_steps
        self.updates = 0
        self.shadow = [tf.Variable(weight, trainable=False) for weight in model.weights]

    def update(self):
        self.updates += 1
        beta = 0. if self.updates == 1 else self.decay * (1 - math.exp(-self.updates / self.warmup_steps))
        for averaged, current in zip(self.shadow, self.model.weights):
            averaged.assign(beta * averaged + (1 - beta) * current if current.trainable else current)
        return beta

    @contextmanager
    def averaged_weights(self):
        original = [weight.numpy() for weight in self.model.weights]
        try:
            for current, averaged in zip(self.model.weights, self.shadow):
                current.assign(averaged)
            yield self.model
        finally:
            for current, previous in zip(self.model.weights, original):
                current.assign(previous)
