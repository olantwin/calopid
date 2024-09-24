#!/usr/bin/env sh
#
# shellcheck disable=SC1091
. /cvmfs/sft.cern.ch/lcg/views/LCG_105a_swan/x86_64-el9-gcc13-opt/setup.sh

DATAPATH=/eos/experiment/sndlhc/users/olantwin/advsnd/2024/09/nu12/Default/
MODELPATH=/eos/user/o/olantwin/SWAN_projects/calopid/
INPUTFILE=$4
MODELFILE=$1
BATCH_SIZE=$2
TARGET=$3
WORK_DIR=/afs/cern.ch/user/o/olantwin/work/advsnd/calopid/

xrdcp root://eospublic.cern.ch/"$DATAPATH/$INPUTFILE" .
xrdcp root://eosuser.cern.ch/"$MODELPATH/$MODELFILE" .

python $WORK_DIR/predict_cnn.py --model "$MODELFILE" -b "$BATCH_SIZE" --data "$INPUTFILE" --target "$TARGET"

cp ./*.csv $MODELPATH
