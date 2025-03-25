"""Custom loss functions for CNN training."""

import tensorflow as tf


def normalised_mae(y_true, y_pred):
    """Normalsed mean average error."""
    return tf.keras.ops.mean(tf.keras.ops.abs((y_pred - y_true) / y_true))


def normalised_rmse(y_true, y_pred):
    """Normalised root mean squared error."""
    return tf.sqrt(tf.keras.ops.mean(((y_pred - y_true) / y_true) ** 2))


def normalised_mse(y_true, y_pred):
    """Normalised mean squared error."""
    return tf.keras.ops.mean(((y_pred - y_true) / y_true) ** 2)
