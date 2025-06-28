"""Evaluate CVTs."""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import uproot
from scipy.optimize import basinhopping
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision, models
from tensorflow.keras.models import load_model

from config import (
    frac_hadr,
    frac_muon,
    nu_e_yield,
    nu_mu_yield,
    nu_tau_yield,
    tensor_spec_mufilter,
    tensor_spec_target,
)
from preprocessing import reshape_data

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

muonic = True
hadronic = not muonic

le = LabelEncoder()

plt.rcParams["font.size"] = 14
plt.rcParams["axes.formatter.limits"] = -5, 4
plt.rcParams["figure.figsize"] = 6, 4
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def event_generator(filename, target, le):
    """Generate events for use by Keras from file."""
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
    """Evaluate a trained Keras CVT model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        help="""Keras model to load.""",
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
        "--target",
        help="""Target variable for inference.""",
        default="nu_flavour",
        choices=["nu_flavour", "nu_energy"],
    )
    args = parser.parse_args()

    model_name = args.model
    le = LabelEncoder()
    le.fit([12 if "non-muonic" in args.data else 14, 16])

    events = uproot.open(args.data + ":df")
    n_events = events.num_entries

    ds_test = (
        tf.data.Dataset.from_generator(
            (lambda: event_generator(args.data + ":df", args.target, le)),
            output_signature=(
                tf.TensorSpec(shape=tensor_spec_target, dtype=tf.float16),
                tf.TensorSpec(shape=tensor_spec_mufilter, dtype=tf.float16),
                tf.TensorSpec(shape=(), dtype=tf.float16),
            ),
        )
        .map(reshape_data)
        .apply(tf.data.experimental.assert_cardinality(events.num_entries))
        .take((n_events // args.batch_size) * args.batch_size)
        .batch(args.batch_size)
    )

    keras.config.enable_unsafe_deserialization()
    model = load_model(args.model)

    y_test = le.transform(np.abs(events[args.target]))[
        : ((n_events // args.batch_size) * args.batch_size)
    ]
    y_pred = model.predict(ds_test)
    scaling_muon = np.array([nu_mu_yield, nu_tau_yield * frac_muon]).reshape(-1, 1)
    scaling_hadr = np.array([nu_e_yield, nu_tau_yield * frac_hadr]).reshape(-1, 1)

    def metric(threshold=0.5):
        TN, FP, FN, TP = confusion_matrix(
            y_test, (y_pred >= threshold).astype(int)
        ).ravel()
        signal = TP / (TP + FN) * nu_tau_yield * (frac_muon if muonic else frac_hadr)
        background = FP / (FP + TN) * (nu_mu_yield if muonic else nu_e_yield)
        return -signal / np.sqrt(signal + background)

    res = basinhopping(metric, 0.2)
    optimum_threshold = res.x[0]
    y_pred_bool = (y_pred >= optimum_threshold).astype(int)
    plt.hist(y_pred_bool)
    plt.hist(y_test)

    ### confusion_matrix_balanced ###
    fig, ax = plt.subplots(figsize=(6, 4))
    titles_options = [
        ("Confusion matrix, without normalisation", None),
    ]
    for title, normalize in titles_options:
        scale = 1.0
        if normalize == "scaled":
            normalize = "true"
            scale = scaling_hadr if hadronic else scaling_muon
        disp = ConfusionMatrixDisplay(
            confusion_matrix(
                le.inverse_transform(y_test.ravel()),
                le.inverse_transform(y_pred_bool.ravel()),
                normalize=normalize,
            )
            * scale,
            display_labels=[r"$\nu_e$" if hadronic else r"$\nu_\mu$", r"$\nu_\tau$"],
        )
        disp.plot(cmap=plt.cm.Blues, ax=ax)

    plt.text(
        0.785,
        1.02,
        "AdvSND",
        fontweight="bold",
        fontfamily="sans-serif",
        fontsize=16,
        transform=ax.transAxes,
        usetex=False,
    )

    plt.savefig(f"confusion_matrix_balanced_{model_name}.png")
    plt.savefig(f"confusion_matrix_balanced_{model_name}.pdf")

    ### confusion_matrix_normalised ###
    fig, ax = plt.subplots(figsize=(6, 4))

    titles_options = [
        ("Normalised confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        scale = 1.0
        if normalize == "scaled":
            normalize = "true"
            scale = scaling_hadr if hadronic else scaling_muon
        disp = ConfusionMatrixDisplay(
            confusion_matrix(
                le.inverse_transform(y_test.ravel()),
                le.inverse_transform(y_pred_bool.ravel()),
                normalize=normalize,
            )
            * scale,
            display_labels=[r"$\nu_e$" if hadronic else r"$\nu_\mu$", r"$\nu_\tau$"],
        )
        disp.plot(cmap=plt.cm.Blues, ax=ax)

    plt.text(
        0.785,
        1.02,
        "AdvSND",
        fontweight="bold",
        fontfamily="sans-serif",
        fontsize=16,
        transform=ax.transAxes,
        usetex=False,
    )

    plt.savefig(f"confusion_matrix_normalised_{model_name}.png")
    plt.savefig(f"confusion_matrix_normalised_{model_name}.pdf")

    ### confusion_matrix_scaled ###
    fig, ax = plt.subplots(figsize=(6, 4))

    titles_options = [
        # ("Confusion matrix, without normalisation", None),
        # ("Normalised confusion matrix", "true"),
        ("Scaled confusion matrix", "scaled"),
    ]
    for title, normalize in titles_options:
        scale = 1.0
        if normalize == "scaled":
            normalize = "true"
            scale = scaling_hadr if hadronic else scaling_muon
        disp = ConfusionMatrixDisplay(
            confusion_matrix(
                le.inverse_transform(y_test.ravel()),
                le.inverse_transform(y_pred_bool.ravel()),
                normalize=normalize,
            )
            * scale,
            display_labels=[r"$\nu_e$" if hadronic else r"$\nu_\mu$", r"$\nu_\tau$"],
        )
        disp.plot(cmap=plt.cm.Blues, ax=ax)

    plt.text(
        0.785,
        1.02,
        "AdvSND",
        fontweight="bold",
        fontfamily="sans-serif",
        fontsize=16,
        transform=ax.transAxes,
        usetex=False,
    )

    plt.savefig(f"confusion_matrix_scaled_{model_name}.png")
    plt.savefig(f"confusion_matrix_scaled_{model_name}.pdf")

    ### roc_curve ###
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(
        y_test, y_pred, plot_chance_level=True, ax=ax, figure=fig
    )

    plt.text(
        0.785,
        1.02,
        "AdvSND",
        fontweight="bold",
        fontfamily="sans-serif",
        fontsize=16,
        transform=ax.transAxes,
        usetex=False,
    )

    plt.savefig(f"ROC_curve_{model_name}.png")
    plt.savefig(f"ROC_curve_{model_name}.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
