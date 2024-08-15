ADVSNDBUILD_DIR=/afs/cern.ch/user/o/olantwin/SND
source /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/setUp.sh
source $ADVSNDBUILD_DIR/nusim_automation/env_cleanup.sh
source $ADVSNDBUILD_DIR/advsnd_minimal_240724.sh
NEUTRINO=12
EVENTGENLIST=CCDIS
OUTPUT_PREFIX=2024/07/nu$NEUTRINO/$EVENTGENLIST
EOSSERVER=root://eospublic.cern.ch/
export FAIRSHIP_ROOT=$ADVSNDSW_ROOT
export FAIRSHIP=$FAIRSHIP_ROOT
export SNDSW_ROOT=$ADVSNDSW_ROOT
