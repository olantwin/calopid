#!/usr/bin/env bash
set -o errexit -o pipefail -o noclobber

# Set up SND environment
WORKDIR=/afs/cern.ch/work/o/olantwin/advsnd/calopid
source "${WORKDIR}"/config.sh

set -o nounset

ProcId=$1
LSB_JOBINDEX=$((ProcId+1))

INPUTFILE=sndLHC.Genie-TGeant4_dig.root
IMAGEFILE=images_1000_"${LSB_JOBINDEX}".npy
FEATUREFILE=features_CNN_1000_"${LSB_JOBINDEX}".csv

OUTPUTDIR=/eos/experiment/sndlhc/users/olantwin/advsnd/"${OUTPUT_PREFIX}"/

set -x

if xrdfs "${EOSSERVER}" stat "${OUTPUTDIR}"/"${IMAGEFILE}" && \
    xrdfs "${EOSSERVER}" stat "${OUTPUTDIR}"/"${FEATUREFILE}"; then
    echo "Target exists, nothing to do."
    exit 0
fi

INDICES=$(seq $(( ProcId * 10 + 1 )) $(( ProcId * 10 + 10 )))
INPUTFILES=$(for i in ${INDICES}; do echo "${OUTPUTDIR}"/"${i}"/"${INPUTFILE}"; done)

# shellcheck disable=SC2086,SC2250
python "${WORKDIR}"/image_creator.py -j 4 --inputfiles $INPUTFILES
mv images_1000.npy "${IMAGEFILE}"
mv features_CNN_1000.csv "${FEATUREFILE}"

xrdcp "${IMAGEFILE}" "${EOSSERVER}"/"${OUTPUTDIR}"/"${IMAGEFILE}"
xrdcp "${FEATUREFILE}" "${EOSSERVER}"/"${OUTPUTDIR}"/"${FEATUREFILE}"
