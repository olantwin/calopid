#!/usr/bin/env sh
#
# shellcheck disable=SC1091
. /cvmfs/sft.cern.ch/lcg/views/LCG_106a/x86_64-el9-gcc11-opt/setup.sh

export PYTHONPATH=/eos/user/g/grassim/.local/lib/python3.11/site-packages:$PYTHONPATH

DATAPATH=/eos/user/g/grassim/SWAN_projects/noexcave/
MODELPATH="$DATAPATH"
INPUTFILE=$3
MODELFILE=$1
BATCH_SIZE=$2
WORK_DIR=/eos/user/g/grassim/SWAN_projects/noexcave/

xrdcp root://eosuser.cern.ch/"$DATAPATH/$INPUTFILE" .
xrdcp root://eosuser.cern.ch/"$MODELPATH/$MODELFILE" .

python $WORK_DIR/evaluate_cvt_cls.py --model "$MODELFILE" -b "$BATCH_SIZE" --data "$INPUTFILE"

cp ./*.png $MODELPATH
cp ./*.pdf $MODELPATH
