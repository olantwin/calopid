# shellcheck shell=bash
TAG=2024/09
ADVSNDBUILD_DIR=/afs/cern.ch/user/o/olantwin/SND
# shellcheck disable=SC1091
source /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/setUp.sh
# shellcheck disable=SC1091
source $ADVSNDBUILD_DIR/nusim_automation/env_cleanup.sh
# shellcheck disable=SC1091
source $ADVSNDBUILD_DIR/advsnd_minimal_240821.sh
NEUTRINO=12
EVENTGENLIST=Default
export OUTPUT_PREFIX=$TAG/nu$NEUTRINO/$EVENTGENLIST
export EOSSERVER=root://eospublic.cern.ch/
export FAIRSHIP_ROOT=$ADVSNDSW_ROOT
export FAIRSHIP=$FAIRSHIP_ROOT
export SNDSW_ROOT=$ADVSNDSW_ROOT
export GEOFILE="/afs/cern.ch/user/o/olantwin/SND/nusim_automation/geofile.advsnd_minimal_v4.gdml"
