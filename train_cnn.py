#!/usr/bin/env python
# coding: utf-8
"""Train CNNs."""

import argparse
import logging
import uproot
import pandas as pd
import numpy as np
import os.path
import tensorflow as tf
from tensorflow.keras.models import load_model


def event_generator(filename):
    """Generate events for CNN training."""
    with uproot.open(filename) as tree:
        for batch in tree.iterate(step_size="1MB", library="np"):
            hitmaps, start_z, nu_energy, energy_dep_target, _ = batch.values()
            for i in range(hitmaps.shape[0]):
                yield hitmaps.astype(np.float16)[i], start_z[i]


def reshape_data(hitmaps, truth):
    """Reshape inpout data for CNN."""
    hitmaps_v = hitmaps[:, ::2]
    hitmaps_h = hitmaps[:, 1::2]
    hitmaps_v_T = tf.transpose(hitmaps_v)
    hitmaps_h_T = tf.transpose(hitmaps_h)
    X_v = tf.expand_dims(hitmaps_v_T, 2)
    X_h = tf.expand_dims(hitmaps_h_T, 2)
    return (X_v, X_h), truth


def main():
    """Train a pre-built Keras CNN model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        help="""Keras model to load.""",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        help="""Number of epochs to train for to use.""",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        help="""Number of samples per batch.""",
        required=True,
    )
    parser.add_argument(
        "--data",
        help="""Training dataset to use.""" """Supports retieval via XRootD.""",
        required=True,
    )
    args = parser.parse_args()

    events = uproot.open(args.data + ":df")

    ds_train = (
        tf.data.Dataset.from_generator(
            (lambda: event_generator(args.data + ":df")),
            output_signature=(
                tf.TensorSpec(shape=(3072, 200), dtype=tf.float16),
                tf.TensorSpec(shape=(), dtype=tf.float64),
            ),
        )
        .map(reshape_data)
        .apply(tf.data.experimental.assert_cardinality(events.num_entries))
        .batch(args.batch_size)
    )

    model = load_model(args.model)

    fit_result = model.fit(
        ds_train.prefetch(tf.data.AUTOTUNE),
        epochs=args.epochs,
    )

    model_name = "_".join(os.path.split(args.model)[-1].split("_")[:3])
    n_events = events.num_entries
    epochs = (
        int(os.path.split(args.model)[-1].split("_")[4][1:].split(".")[0]) + args.epochs
    )
    model.save(f"{model_name}_n{n_events}_e{epochs}.keras")
    history = pd.DataFrame(fit_result.history)
    history.to_csv(f"history_{model_name}_n{n_events}_e{epochs}.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
