#!/usr/bin/env python
# coding: utf-8
"""Create 1d images and features for use in CNNs."""

import argparse
import logging
import ROOT
import pandas as pd
import numpy as np


def main():
    """Create images and features for use in CNNs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputfiles",
        nargs="+",
        help="""List of input files to use."""
        """Supports retrieving file from EOS via the XRootD protocol.""",
        required=True,
    )
    parser.add_argument(
        "-j",
        "--num_cpu",
        default=1,
        type=int,
        help="""Number of threads to use.""",
    )
    args = parser.parse_args()
    ROOT.EnableImplicitMT(args.num_cpu)

    df = ROOT.ROOT.RDataFrame("cbmsim", args.inputfiles)

    df = df.Filter("Digi_AdvTargetHits.GetEntries() != 0")
    count = df.Count()

    ROOT.gInterpreter.ProcessLine('#include "ShipMCTrack.h"')
    ROOT.gInterpreter.ProcessLine('#include "AdvTargetHit.h"')

    ROOT.gInterpreter.Declare(
        """
    int station_from_id(int id) {
        return id >>17;
    }
    """
    )
    ROOT.gInterpreter.Declare(
        """
    int plane_from_id(int id) {
        return (id >> 16) % 2;
    }
    """
    )

    df = (
        df.Define("start_z", "dynamic_cast<ShipMCTrack*>(MCTrack[1])->GetStartZ()")
        .Define("stations", "Map(Digi_AdvTargetHits.fDetectorID, station_from_id)")
        .Define("planes", "Map(Digi_AdvTargetHits.fDetectorID, plane_from_id)")
    )

    col_names = [
        "start_z",
        "stations",
        "planes",
    ]

    cols = df.AsNumpy(col_names)
    n_events = count.GetValue()

    hitmaps = np.zeros((n_events, 200))

    for event in range(n_events):
        stations = np.array(cols["stations"][event], dtype=int)
        planes = np.array(cols["planes"][event], dtype=int)
        index = 2 * stations + planes
        unique, counts = np.unique(index, return_counts=True)
        for i, c in zip(unique, counts):
            hitmaps[event, i] = c
        del index
        del planes
        del stations
        del unique
        del counts

    np.save(f"images_1d_{n_events}.npy", hitmaps)

    pandas_df = pd.DataFrame(cols)
    pandas_df.pop("stations")
    pandas_df.pop("planes")

    pandas_df.to_csv(f"features_CNN_1d_{n_events}.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
