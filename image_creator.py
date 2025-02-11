#!/usr/bin/env python
# coding: utf-8
"""Create images and features for use in CNNs."""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import ROOT
import uproot
from tqdm import tqdm


def main():
    """Create images and features for use in CNNs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputfiles",
        nargs="+",
        help="""List of input files to use.\n"""
        """Supports retrieving file from EOS via the XRootD protocol.""",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--outputfile",
        help="""Output file to write to.""",
        required=True,
    )
    parser.add_argument(
        "-j",
        "--num_cpu",
        default=1,
        type=int,
        help="""Number of threads to use.""",
    )
    parser.add_argument(
        "--saturation",
        default=1,
        type=int,
        help="""Saturation threshold, defaults to digital →1.""",
    )
    parser.add_argument(
        "--new_geo",
        action="store_true",
        help="""Use channel mapping etc. for new no-excavation geometry.""",
    )
    parser.add_argument(
        "--muonic",
        action="store_true",
        help="""Select muonic events for ν_τ.""",
    )
    parser.add_argument(
        "--hadronic",
        action="store_true",
        help="""Select non-muonic events for ν_τ.""",
    )
    parser.add_argument(
        "--CC",
        action="store_true",
        help="""Select CC events for neutrinos.""",
    )
    parser.add_argument(
        "--NC",
        action="store_true",
        help="""Select NC events for neutrinos.""",
    )
    parser.add_argument(
        "--fiducial",
        action="store_true",
        help="""Cut on fiducial volume.""",
    )
    parser.add_argument(
        "--plot_events",
        action="store_true",
        help="""Plot events and save them to file.""",
    )
    args = parser.parse_args()
    ROOT.EnableImplicitMT(args.num_cpu)

    ROOT.gInterpreter.ProcessLine('#include "ShipMCTrack.h"')
    ROOT.gInterpreter.ProcessLine('#include "AdvTargetHit.h"')
    ROOT.gInterpreter.ProcessLine('#include "AdvMuFilterHit.h"')
    ROOT.gInterpreter.ProcessLine('#include "Hit2MCPoints.h"')

    df = ROOT.ROOT.RDataFrame("cbmsim", args.inputfiles)
    # ROOT.ROOT.RDF.Experimental.AddProgressBar(df)  # Only available in 6.30+

    df = df.Filter(
        "Digi_AdvMuFilterHits.GetEntries() || Digi_AdvTargetHits.GetEntries()",
        "Preselection",
    )
    ROOT.gInterpreter.Declare(
        """
        int station_from_id(int id) {
        return id >>17;
        }
        """
    )
    ROOT.gInterpreter.Declare(
        """
        int strip_from_id_new(int id) {
        return (id) % 10000;
        }
        """
    )
    ROOT.gInterpreter.Declare(
        """
        int detector_from_id(int id) {
        return (id - strip_from_id_new(id)) / 10000;
        }
        """
    )
    ROOT.gInterpreter.Declare(
        """
        int station_from_id_new(int id) {
        return (detector_from_id(id) / 10) - 1;
        }
        """
    )
    ROOT.gInterpreter.Declare(
        """
        int column_from_id(int id) {
        return (id >> 11) % 4;
        }
        """
    )
    ROOT.gInterpreter.Declare(
        """
        int sensor_from_id(int id) {
        return (id >> 10) % 2;
        }
        """
    )
    ROOT.gInterpreter.Declare(
        """
        int strip_from_id(int id) {
        return (id) % 1024;
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
    ROOT.gInterpreter.Declare(
        """
        int plane_from_id_new(int id) {
        return detector_from_id(id) % 10;
        }
        """
    )
    ROOT.gInterpreter.Declare(
        """
        int is_charged_lepton(int id) {
        return (abs(id) == 11) || (abs(id) == 13) || (abs(id) == 15);
        }
        """
    )
    ROOT.gInterpreter.Declare(
        """
        ROOT::RVec<std::unordered_map<int, float>> wlist(Hit2MCPoints* link, ROOT::RVec<int> ids) {
            ROOT::RVec<std::unordered_map<int, float>> wlists{};
            for (auto&& id : ids) {
                wlists.push_back(link->wList(id));
            }
            return wlists;
        }
        """
    )
    ROOT.gInterpreter.Declare(
        """
        int points_from_weights(std::unordered_map<int, float> weights) {
            return weights.size();
        }
        """
    )
    ROOT.gInterpreter.Declare(f"const int SATURATION = {args.saturation};")
    ROOT.gInterpreter.Declare(
        """
        int apply_saturation(int points) {
            return min(points, SATURATION);
        }
        """
    )
    # TODO simplify? Careful, very fragile!
    ROOT.gInterpreter.Declare(
        """
        int index_from_id(int id) {
            int columns_mufilter = column_from_id(id);
            int sensors_mufilter = sensor_from_id(id);
            int strips_mufilter = strip_from_id(id);
            return (
            (
                2 * columns_mufilter
                + abs((1 - (2 * sensors_mufilter)) * columns_mufilter) % 2
                + sensors_mufilter
                - 2 * (sensors_mufilter * columns_mufilter % 2)
                + columns_mufilter % 2
                + 2
                - 2 * (columns_mufilter + sensors_mufilter >= 1)
            )
            * 768
            + pow(-1, columns_mufilter) * strips_mufilter
            - 2 * strips_mufilter * (columns_mufilter == 0)
            - 1 * (columns_mufilter % 2)
            - 1 * (columns_mufilter == 0)
            );
        }
        """
    )
    ROOT.gInterpreter.Declare(
        """
        bool is_muonic(const TClonesArray& tracks) {
             for (auto* track: tracks) {
                 auto* particle = dynamic_cast<ShipMCTrack*>(track);
                 if (particle->GetMotherId() == 1) {
                     if (abs(particle->GetPdgCode()) == 13) {
                         return true;
                     }
                 }
             }
             return false;
        }
        """
    )
    ROOT.gInterpreter.Declare(
        """
        bool fiducial_cut(double x, double y, double z) {
             if (x < -30 || x > -10)
                  return false;
             if (y < 35 || y > 55)
                  return false;
             if (z < -20 || z > 20)
                  return false;
             return true;
        }
        """
    )

    df = (
        df.Define("start_x", "dynamic_cast<ShipMCTrack*>(MCTrack[1])->GetStartX()")
        .Define("start_y", "dynamic_cast<ShipMCTrack*>(MCTrack[1])->GetStartY()")
        .Define("start_z", "dynamic_cast<ShipMCTrack*>(MCTrack[1])->GetStartZ()")
        .Define("is_fiducial", "fiducial_cut(start_x, start_y, start_z)")
        .Define("nu_energy", "dynamic_cast<ShipMCTrack*>(MCTrack[0])->GetEnergy()")
        .Define("nu_flavour", "dynamic_cast<ShipMCTrack*>(MCTrack[0])->GetPdgCode()")
        .Define(
            "is_cc",
            "is_charged_lepton(dynamic_cast<ShipMCTrack*>(MCTrack[1])->GetPdgCode())",
        )
        .Define(
            "is_nc",
            "!is_cc",
        )
        .Define("muonic", "is_muonic(MCTrack)")
        .Define("non_muonic", "!muonic")
    )
    if args.fiducial:
        df = df.Filter("is_fiducial", "Fiducial volume cut")
    if args.CC:
        df = df.Filter("is_cc", "Only CC")
    elif args.NC:
        df = df.Filter("is_nc", "Only NC")
    if args.muonic:
        df = df.Filter("muonic", "Only muonic")
    elif args.hadronic:
        df = df.Filter("non_muonic", "Everything but muonic")
    df = (
        df.Define(
            "lepton_energy", "dynamic_cast<ShipMCTrack*>(MCTrack[1])->GetEnergy()"
        )  # TODO not reconstructible in NC case
        .Define("hadron_energy", "nu_energy - lepton_energy")
        .Define("energy_dep_target", "Sum(AdvTargetPoint.fELoss)")
        .Define("energy_dep_mufilter", "Sum(AdvMuFilterPoint.fELoss)")
        .Define(
            "link_target", "dynamic_cast<Hit2MCPoints*>(Digi_AdvTargetHits2MCPoints[0])"
        )
        .Define("weights_target", "wlist(link_target, Digi_AdvTargetHits.fDetectorID)")
        .Define("points_per_hit_target", "Map(weights_target, points_from_weights)")
        .Define(
            "saturated_points_per_hit_target",
            "Map(points_per_hit_target, apply_saturation)",
        )
        .Define(
            "link_mufilter",
            "dynamic_cast<Hit2MCPoints*>(Digi_AdvMuFilterHits2MCPoints[0])",
        )
        .Define(
            "weights_mufilter", "wlist(link_mufilter, Digi_AdvMuFilterHits.fDetectorID)"
        )
        .Define("points_per_hit_mufilter", "Map(weights_mufilter, points_from_weights)")
        .Define(
            "saturated_points_per_hit_mufilter",
            "Map(points_per_hit_mufilter, apply_saturation)",
        )
    )
    df = (
        (
            df.Define(
                "stations", "Map(Digi_AdvTargetHits.fDetectorID, station_from_id_new)"
            )
            .Define("strips", "Map(Digi_AdvTargetHits.fDetectorID, strip_from_id_new)")
            .Define("planes", "Map(Digi_AdvTargetHits.fDetectorID, plane_from_id_new)")
            .Define(
                "stations_mufilter",
                "Map(Digi_AdvMuFilterHits.fDetectorID, station_from_id_new)",
            )
            .Define(
                "strips_mufilter",
                "Map(Digi_AdvMuFilterHits.fDetectorID, strip_from_id_new)",
            )
            .Define(
                "planes_mufilter",
                "Map(Digi_AdvMuFilterHits.fDetectorID, plane_from_id_new)",
            )
            .Define(
                "indices",
                "strips",
            )
            .Define("indices_mufilter", "strips_mufilter")
        )
        if args.new_geo
        else (
            df.Define(
                "stations", "Map(Digi_AdvTargetHits.fDetectorID, station_from_id)"
            )
            .Define("columns", "Map(Digi_AdvTargetHits.fDetectorID, column_from_id)")
            .Define("sensors", "Map(Digi_AdvTargetHits.fDetectorID, sensor_from_id)")
            .Define("strips", "Map(Digi_AdvTargetHits.fDetectorID, strip_from_id)")
            .Define("planes", "Map(Digi_AdvTargetHits.fDetectorID, plane_from_id)")
            .Define(
                "stations_mufilter",
                "Map(Digi_AdvMuFilterHits.fDetectorID, station_from_id)",
            )
            .Define(
                "columns_mufilter",
                "Map(Digi_AdvMuFilterHits.fDetectorID, column_from_id)",
            )
            .Define(
                "sensors_mufilter",
                "Map(Digi_AdvMuFilterHits.fDetectorID, sensor_from_id)",
            )
            .Define(
                "strips_mufilter",
                "Map(Digi_AdvMuFilterHits.fDetectorID, strip_from_id)",
            )
            .Define(
                "planes_mufilter",
                "Map(Digi_AdvMuFilterHits.fDetectorID, plane_from_id)",
            )
            .Define(
                "indices",
                "(4 * columns + sensors - 2 * columns * sensors) * 768 + pow(-1, columns) * strips - 1 * columns",
            )
            .Define(
                "indices_mufilter",
                "Map(Digi_AdvMuFilterHits.fDetectorID, index_from_id)",
            )
        )
    )

    report = df.Report()

    col_names = {
        "start_x",
        "start_y",
        "start_z",
        "nu_energy",
        "hadron_energy",
        "lepton_energy",
        "nu_flavour",
        "is_cc",
        "muonic",
        "energy_dep_target",
        "energy_dep_mufilter",
        "indices",
        "stations",
        "planes",
        "indices_mufilter",
        "stations_mufilter",
        "planes_mufilter",
        "saturated_points_per_hit_target",
        "saturated_points_per_hit_mufilter",
    }

    df.Snapshot(
        "df", "temporary.root", col_names
    )  # TODO Use TMatrix to avoid detour via uproot?
    report.Print()

    target_dims = (3279, 116) if args.new_geo else (3072, 200)
    mufilter_dims = (3279, 68) if args.new_geo else (4608, 42)

    events = uproot.open("temporary.root:df")
    # TODO second file for testing?
    outputfile = uproot.recreate(args.outputfile)
    outputfile.mktree(
        "df",
        {
            "X": (">f4", target_dims),
            "X_mufilter": (">f4", mufilter_dims),
            "start_x": ">f8",
            "start_y": ">f8",
            "start_z": ">f8",
            "nu_energy": ">f8",
            "hadron_energy": ">f8",
            "lepton_energy": ">f8",
            "energy_dep_target": ">f8",
            "energy_dep_mufilter": ">f8",
            "nu_flavour": ">i8",
            "is_cc": "bool",
        },
        title="Dataframe for CNN studies",
    )
    t = tqdm(total=events.num_entries)
    for batch in events.iterate(step_size="1MB", library="np"):
        batch_size = batch["is_cc"].shape[0]
        hitmaps = np.zeros((batch_size, *target_dims))
        hitmaps_mufilter = np.zeros((batch_size, *mufilter_dims))
        for i in range(batch_size):
            indices = batch["indices"][i].astype(int)
            stations = batch["stations"][i].astype(int)
            planes = batch["planes"][i].astype(int)
            points = batch["saturated_points_per_hit_target"][i].astype(int)
            hitmaps[i, indices, 2 * stations + planes] = points
            indices = batch["indices_mufilter"][i].astype(int)
            stations = batch["stations_mufilter"][i].astype(int)
            planes = batch["planes_mufilter"][i].astype(int)
            points = batch["saturated_points_per_hit_mufilter"][i].astype(int)
            hitmaps_mufilter[i, indices, 2 * stations + planes] = points
            if args.plot_events:
                plt.subplot(3, 2, 1)
                plt.imshow(hitmaps[i, :, 0:-1:2], aspect="auto")
                plt.subplot(3, 2, 2)
                plt.imshow(hitmaps_mufilter[i, :, 0:-1:2], aspect="auto")
                plt.subplot(3, 2, 3)
                plt.imshow(hitmaps[i, :, 1::2], aspect="auto")
                plt.subplot(3, 2, 4)
                plt.imshow(hitmaps_mufilter[i, :, 1::2], aspect="auto")
                plt.subplot(3, 2, 5)
                plt.imshow(hitmaps[i, :, ::], aspect="auto")
                plt.subplot(3, 2, 6)
                plt.imshow(hitmaps_mufilter[i, :, ::], aspect="auto")
                plt.show()
        outputfile["df"].extend(
            {
                "X": hitmaps.astype(np.float32),
                "X_mufilter": hitmaps_mufilter.astype(np.float32),
                "start_x": batch["start_x"],
                "start_y": batch["start_y"],
                "start_z": batch["start_z"],
                "nu_energy": batch["nu_energy"],
                "hadron_energy": batch["hadron_energy"],
                "lepton_energy": batch["lepton_energy"],
                "energy_dep_target": batch["energy_dep_target"],
                "energy_dep_mufilter": batch["energy_dep_mufilter"],
                "nu_flavour": batch["nu_flavour"],
                "is_cc": batch["is_cc"],
            }
        )
        t.update(batch_size)
    outputfile.close()
    os.remove("temporary.root")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
