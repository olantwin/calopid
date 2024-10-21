#!/usr/bin/env python
# coding: utf-8
"""Train CNNs."""

import argparse
import logging
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
import uproot
from tensorflow.keras.models import load_model


def event_generator(filename, target):
    """Generate events for CNN training."""
    log = "energy" in target
    with uproot.open(filename) as events:
        for batch, report in events.iterate(step_size=1, report=True, library="np"):
            for i in range(batch["X"].shape[0]):
                yield (
                    batch["X"].astype(np.float16)[i],
                    batch["X_mufilter"].astype(np.float16)[i],
                    (np.log(batch[target][i]) if log else batch[target][i]),
                )


@tf.function
def reshape_data(hitmaps, hitmaps_mufilter, truth):
    """Reshape data from hitmaps per subsystem to hitmaps per view."""
    hitmaps_v = hitmaps[:, ::2]
    hitmaps_h = hitmaps[:, 1::2]
    hitmaps_v_T = tf.transpose(hitmaps_v)
    hitmaps_h_T = tf.transpose(hitmaps_h)
    X_v = tf.expand_dims(hitmaps_v_T, 2)
    X_h = tf.expand_dims(hitmaps_h_T, 2)
    hitmaps_v = hitmaps_mufilter[:, ::2]
    hitmaps_h = hitmaps_mufilter[:, 1:10:2]
    hitmaps_v_T = tf.transpose(hitmaps_v)
    hitmaps_h_T = tf.transpose(hitmaps_h)
    X_mufilter_v = tf.expand_dims(hitmaps_v_T, 2)
    X_mufilter_h = tf.expand_dims(hitmaps_h_T, 2)
    return (X_v, X_h, X_mufilter_v, X_mufilter_h), truth


def main():
    """Train a pre-built Keras CNN model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        help="""Keras model to load.""",
        required=True,
    )
    parser.add_argument(
        "--target",
        help="""Target observable.""",
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
        help="""Test dataset to use.""" """Supports retieval via XRootD.""",
        required=True,
    )
    args = parser.parse_args()
    model_name = "_".join(os.path.split(args.model)[-1].split("_")[:5])
    print(f"Predicting using model {model_name}.")
    events = uproot.open(args.data + ":df")
    y_test = events[args.target].array()

    ds_test = (
        tf.data.Dataset.from_generator(
            (lambda: event_generator(args.data + ":df", args.target)),
            output_signature=(
                tf.TensorSpec(shape=(3072, 200), dtype=tf.float16),
                tf.TensorSpec(shape=(4608, 42), dtype=tf.float16),
                tf.TensorSpec(shape=(), dtype=tf.float64),
            ),
        )
        .map(reshape_data)
        .apply(tf.data.experimental.assert_cardinality(events.num_entries))
        .batch(args.batch_size)
    )

    model = load_model(args.model)
    if len(model.inputs) == 2:
        # Old format, target only
        # TODO nonsense results, what's going on?
        ds_test = ds_test.map(lambda x, y: ((x[0], x[1]), y))
    y_pred = model.predict(ds_test)
    if "energy" in args.target:
        y_pred = np.exp(y_pred)

    n_events = events.num_entries
    epochs = int(os.path.split(args.model)[-1].split("_")[6][1:].split(".")[0])
    df = pd.DataFrame(
        {
            f"{args.target}_pred": np.squeeze(y_pred),
            f"{args.target}_test": np.squeeze(y_test),
        }
    )
    df.to_csv(f"{model_name}_n{n_events}_e{epochs}.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
