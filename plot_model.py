#!/usr/bin/env python
# coding: utf-8
"""Plot model architecture."""

import argparse
import logging
import os.path

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model


def main():
    """Plot model architecture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="""File containing the model to plot.""")
    parser.add_argument(
        "-o", "--output_dir", help="""Output directory for plots.""", default="plots"
    )
    args = parser.parse_args()
    model = load_model(args.model)
    model_name = "_".join(os.path.split(args.model)[-1].split("_")[:3])
    model.summary()
    plot_model(model, show_shapes=True, to_file=f"{args.output_dir}/{model_name}.png")
    plot_model(model, show_shapes=True, to_file=f"{args.output_dir}/{model_name}.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
