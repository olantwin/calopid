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
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ReduceLROnPlateau

from config import tensor_spec_target, tensor_spec_mufilter
from preprocessing import reshape_data


def event_generator(filename, target, le):
    with uproot.open(filename) as events:
        for batch, report in events.iterate(step_size=1, report=True, library="np"):
            ys = le.transform(np.abs(batch[target]))
            for i in range(batch["X"].shape[0]):
                yield (
                    batch["X"].astype(np.float16)[i],
                    batch["X_mufilter"].astype(np.float16)[i],
                    ys[i],
                )


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
    parser.add_argument(
        "--regression",
        help="""Whether to perform regression or classification (default).""",
        action="store_true",
    )
    parser.add_argument(
        "--target",
        help="""Target variable for inference.""",
        default="nu_flavour",
        choices=["nu_flavour", "nu_energy"],
    )
    args = parser.parse_args()

    events = uproot.open(args.data + ":df")

    le = LabelEncoder()
    le.fit([12 if "non-muonic" in args.data else 14, 16])

    ds_train = (
        tf.data.Dataset.from_generator(
            (lambda: event_generator(args.data + ":df", args.target, le)),
            output_signature=(
                tf.TensorSpec(shape=tensor_spec_target, dtype=tf.float16),
                tf.TensorSpec(shape=tensor_spec_mufilter, dtype=tf.float16),
                tf.TensorSpec(
                    shape=(), dtype=tf.float64 if args.regression else tf.int64
                ),
            ),
        )
        .map(reshape_data)
        .apply(tf.data.experimental.assert_cardinality(events.num_entries))
        .batch(args.batch_size)
    )

    model = load_model(args.model)

    reduce_lr = ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1
    )

    fit_result = model.fit(
        ds_train.prefetch(tf.data.AUTOTUNE),
        epochs=args.epochs,
        callbacks=[reduce_lr],
    )

    model_name = model.name
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
