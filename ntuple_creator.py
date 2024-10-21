#!/usr/bin/env python
# coding: utf-8
"""Create features for use in MVA algorithms."""

import argparse
import logging

import pandas as pd
import ROOT


def main():
    """Create features for use in MVA algorithms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputfiles",
        nargs="+",
        help="""List of input files to use."""
        """Supports retrieving file from EOS via the XRootD protocol.""",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--outputfile",
        help="""File to write the filtered tree to."""
        """Will be recreated if it already exists.""",
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

    df = df.Filter(
        "Digi_AdvMuFilterHits.GetEntries() || Digi_AdvTargetHits.GetEntries()"
    )
    count = df.Count()

    ROOT.gInterpreter.ProcessLine('#include "ShipMCTrack.h"')
    ROOT.gInterpreter.ProcessLine('#include "AdvTargetHit.h"')
    ROOT.gInterpreter.ProcessLine('#include "AdvMuFilterHit.h"')

    ROOT.gInterpreter.Declare(
        """
    int station_from_id(int id) {
        return id >>17;
    }
    """
    )
    ROOT.gInterpreter.Declare(
        """
    template<typename T>
    ROOT::RVec<T> Deduplicate (ROOT::RVec<T> v){
        std::sort(v.begin(), v.end());
        auto last = std::unique(v.begin(), v.end());
        v.erase(last, v.end());
        return v;
    }
    """
    )

    df = (
        df.Define("start_z", "dynamic_cast<ShipMCTrack*>(MCTrack[1])->GetStartZ()")
        .Define("nu_energy", "dynamic_cast<ShipMCTrack*>(MCTrack[0])->GetEnergy()")
        .Define("energy_dep_target", "Sum(AdvTargetPoint.fELoss)")
        .Define("energy_dep_mufilter", "Sum(AdvMuFilterPoint.fELoss)")
        .Define(
            "target_stations", "Map(Digi_AdvTargetHits.fDetectorID, station_from_id)"
        )
        .Define(
            "mufilter_stations",
            "Map(Digi_AdvMuFilterHits.fDetectorID, station_from_id)",
        )
        .Define("target_n_stations", "Deduplicate(target_stations).size()")
        .Define("mufilter_n_stations", "Deduplicate(mufilter_stations).size()")
        .Define("target_n_hits", "Digi_AdvTargetHits.GetEntries()")
        .Define("mufilter_n_hits", "Digi_AdvMuFilterHits.GetEntries()")
    )

    for i in range(100):
        df = df.Define(
            f"target_n_hits_station_{i}",
            f"std::count(target_stations.begin(), target_stations.end(), {i})",
        )

    for i in range(20):
        df = df.Define(
            f"mufilter_n_hits_station_{i}",
            f"std::count(mufilter_stations.begin(), mufilter_stations.end(), {i})",
        )

    col_names = (
        [
            "start_z",
            "nu_energy",
            "energy_dep_target",
            "energy_dep_mufilter",
            "target_n_hits",
            "target_n_stations",
            "mufilter_n_hits",
            "mufilter_n_stations",
        ]
        + [f"target_n_hits_station_{i}" for i in range(100)]
        + [f"mufilter_n_hits_station_{i}" for i in range(20)]
    )

    cols = df.AsNumpy(col_names)
    n_events = count.GetValue()

    pandas_df = pd.DataFrame(cols)

    pandas_df.to_csv(f"features_{n_events}.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
