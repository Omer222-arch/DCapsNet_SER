"""DCapsNet: 1-D conv feature extractor + primary capsules + digit capsules
with dynamic routing-by-agreement (3 iterations) and margin loss."""
import tensorflow as tf
import keras
from keras import layers

PKG = "dcapsnet"


def squash(s, axis=-1, eps=1e-8):
    sq = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
    return (sq / (1.0 + sq)) * s / tf.sqrt(sq + eps)


@keras.saving.register_keras_serializable(package=PKG)
class DigitCaps(layers.Layer):
    """One 16-D capsule per emotion class, fed by all primary capsules."""

    def __init__(self, num_caps, dim_caps=16, routings=3, **kwargs):
        super().__init__(**kwargs)
        self.num_caps, self.dim_caps, self.routings = num_caps, dim_caps, routings

    def build(self, input_shape):
        in_caps, in_dim = input_shape[1], input_shape[2]
        self.W = self.add_weight(
            name="W", shape=(in_caps, self.num_caps, self.dim_caps, in_dim),
            initializer="glorot_uniform", trainable=True)

    def call(self, u):
        # prediction vectors u_hat: (batch, in_caps, num_caps, dim_caps)
        u_hat = tf.einsum("ijdl,bil->bijd", self.W, u)
        b = tf.zeros_like(u_hat[..., 0])                 # routing logits
        for r in range(self.routings):
            c = tf.nn.softmax(b, axis=2)                 # coupling coefficients
            s = tf.reduce_sum(c[..., None] * u_hat, axis=1)
            v = squash(s)                                # (batch, num_caps, dim_caps)
            if r < self.routings - 1:
                b = b + tf.reduce_sum(u_hat * v[:, None, :, :], axis=-1)
        return v

    def get_config(self):
        cfg = super().get_config()
        cfg.update(num_caps=self.num_caps, dim_caps=self.dim_caps, routings=self.routings)
        return cfg


@keras.saving.register_keras_serializable(package=PKG)
class CapsuleLength(layers.Layer):
    """L2 norm of each capsule vector = class score."""

    def call(self, v):
        return tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1) + 1e-8)


@keras.saving.register_keras_serializable(package=PKG)
def margin_loss(y_true, y_pred, m_pos=0.9, m_neg=0.1, lam=0.5):
    left = y_true * tf.square(tf.maximum(0.0, m_pos - y_pred))
    right = lam * (1.0 - y_true) * tf.square(tf.maximum(0.0, y_pred - m_neg))
    return tf.reduce_mean(tf.reduce_sum(left + right, axis=-1))


def build_dcapsnet(n_classes, n_features=217, dropout=0.1):
    inp = keras.Input(shape=(n_features,), name="input")
    x = layers.Reshape((n_features, 1))(inp)
    # (filters, kernel, stride) per paper Table 5 / Sec. 3.5
    for i, (f, k, s) in enumerate([(32, 9, 1), (64, 9, 1), (128, 7, 1), (256, 7, 2)], 1):
        x = layers.Conv1D(f, k, strides=s, padding="valid", name=f"conv{i}")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ELU()(x)
        x = layers.Dropout(dropout)(x)
    # primary capsules: 1-D conv (128 filters, k=5, s=2) + ReLU, regrouped into 8-D vectors
    x = layers.Conv1D(128, 5, strides=2, padding="valid", activation="relu",
                      name="primary_caps")(x)            # (None, 46, 128)
    x = layers.Reshape((-1, 8))(x)                       # (None, 736, 8)
    v = DigitCaps(n_classes, 16, routings=3, name="digit_caps")(x)
    out = CapsuleLength(name="capsule_lengths")(v)
    return keras.Model(inp, out, name="DCapsNet")


if __name__ == "__main__":
    build_dcapsnet(8).summary()
