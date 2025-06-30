#!/usr/bin/env sh
#
# shellcheck disable=SC1091
. /cvmfs/sft.cern.ch/lcg/views/LCG_106a/x86_64-el9-gcc11-opt/setup.sh

export PYTHONPATH=/eos/user/g/grassim/.local/lib/python3.11/site-packages:"$PYTHONPATH"

DATAPATH=/eos/user/g/grassim/SWAN_projects/noexcave/root_files/
MODELPATH=/eos/user/g/grassim/SWAN_projects/noexcave/model_zoo/
INPUTFILE=$4
MODELFILE=$1
BATCH_SIZE=$2
EPOCHS=$3
WORK_DIR=/eos/user/g/grassim/SWAN_projects/noexcave/

xrdcp root://eosuser.cern.ch/"$DATAPATH/$INPUTFILE" .
xrdcp root://eosuser.cern.ch/"$MODELPATH/$MODELFILE" .

python $WORK_DIR/train_cvt.py --model "$MODELFILE" -b "$BATCH_SIZE" -n "$EPOCHS" --data "$INPUTFILE"

rm ./"$MODELFILE"
cp ./*.keras $MODELPATH
cp ./*.csv $MODELPATH
