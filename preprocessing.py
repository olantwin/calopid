"""Preprocessing and IO functions."""

import tensorflow as tf


@tf.function
def reshape_data(hitmaps, hitmaps_mufilter_h, truth):
    """Reshape data from hitmaps per subsystem to hitmaps per view."""
    hitmaps_v = hitmaps[:, ::2]
    hitmaps_h = hitmaps[:, 1::2]
    hitmaps_v_T = tf.transpose(hitmaps_v)
    hitmaps_h_T = tf.transpose(hitmaps_h)
    X_v = tf.expand_dims(hitmaps_v_T, 2)
    X_h = tf.expand_dims(hitmaps_h_T, 2)
    hitmaps_h_T = tf.transpose(hitmaps_mufilter_h)
    X_mufilter_h = tf.expand_dims(hitmaps_h_T, 2)
    return (X_v, X_h, X_mufilter_h), truth
