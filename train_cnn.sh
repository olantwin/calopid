#!/usr/bin/env sh
#
# shellcheck disable=SC1091
. /cvmfs/sft.cern.ch/lcg/views/LCG_105a_swan/x86_64-el9-gcc13-opt/setup.sh

DATAPATH=/eos/user/o/olantwin/SWAN_projects/calopid/
MODELPATH="$DATAPATH"
INPUTFILE=CNN_dataframe_80000_train.root
MODELFILE=$1
BATCH_SIZE=$2
EPOCHS=$3
WORK_DIR=/afs/cern.ch/user/o/olantwin/work/advsnd/calopid/

xrdcp root://eosuser.cern.ch/"$DATAPATH/$INPUTFILE" .
xrdcp root://eosuser.cern.ch/"$MODELPATH/$MODELFILE" .

python $WORK_DIR/train_cnn.py --model "$MODELFILE" -b "$BATCH_SIZE" -n "$EPOCHS" --data "$INPUTFILE"

rm ./"$MODELFILE"
cp ./*.keras $MODELPATH
cp ./*.csv $MODELPATH
