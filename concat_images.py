#!/usr/bin/env python3
"""Concatenate image files into single array/file."""

import argparse
import logging
import numpy as np


def main():
    """Create images and features for use in CNNs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputfiles",
        nargs="+",
        help="""List of input files to use.""",
        required=True,
    )
    parser.add_argument(
        "--outputfile",
        help="""Output file to write to.""",
        required=True,
    )
    args = parser.parse_args()
    all_arrays = []
    for inputfile in args.inputfiles:
        all_arrays.append(np.load(inputfile))
    all_arrays = np.concatenate(all_arrays)
    np.save(args.outputfile, all_arrays)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
