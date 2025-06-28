#!/usr/bin/env python
# coding: utf-8
"""Train CVTs."""

import argparse
import logging

import numpy as np
import pandas as pd
import tensorflow
import tensorflow as tf
import uproot
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from tensorflow.keras.models import load_model

from config import tensor_spec_mufilter, tensor_spec_target
from preprocessing import reshape_data


def event_generator(filename, target, le):
    """Generate events for use by Keras from file."""
    with uproot.open(filename) as events:
        for batch, report in events.iterate(step_size=1, report=True, library="np"):
            ys = (
                le.transform(np.abs(batch[target]))
                if target == "nu_flavour"
                else np.log(batch[target])
            )
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
        "--batch_size",
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

    if "energy" in args.model:
        args.target = "nu_energy"
        args.regression = True

    le = LabelEncoder()
    le.fit([12 if "non-muonic" in args.data else 14, 16])

    n_events = events.num_entries

    ds_train = (
        tf.data.Dataset.from_generator(
            (lambda: event_generator(args.data + ":df", args.target, le)),
            output_signature=(
                tf.TensorSpec(shape=tensor_spec_target, dtype=tf.float16),
                tf.TensorSpec(shape=tensor_spec_mufilter, dtype=tf.float16),
                tf.TensorSpec(
                    shape=(), dtype=tf.float16 if args.regression else tf.int64
                ),
            ),
        )
        .map(reshape_data)
        .apply(tf.data.experimental.assert_cardinality(events.num_entries))
        # .take((n_events//args.batch_size)*args.batch_size)
        .take(256)
        .batch(args.batch_size)
    )

    keras.config.enable_unsafe_deserialization()
    model = load_model(args.model)

    model.compile(
        optimizer="Adam",
        loss=tensorflow.keras.losses.BinaryFocalCrossentropy(from_logits=True),
        metrics=[
            tensorflow.keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
            tensorflow.keras.metrics.AUC(from_logits=True),
        ],
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1
    )

    early_stopping = EarlyStopping(
        monitor="loss", patience=18, verbose=1, restore_best_weights=True
    )

    fit_result = model.fit(
        ds_train.prefetch(tf.data.AUTOTUNE),
        epochs=args.epochs,
        callbacks=[reduce_lr, early_stopping],
    )

    model_name = model.name
    epochs = args.epochs + int(args.model.split("_")[-1].split(".")[0][1:])
    batch_size = args.batch_size
    model.save(f"{model_name}_n{n_events}_b{batch_size}_e{epochs}.keras")
    history = pd.DataFrame(fit_result.history)
    history.to_csv(f"history_{model_name}_n{n_events}_b{batch_size}_e{epochs}.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
