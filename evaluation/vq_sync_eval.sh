#!/bin/bash

eval_path=$1
filelist=$2
sync_root=$3
gt_v_root=$4
bbx_root=$5
gpu=$6

echo "Evaluating Synchronization...."
echo
python sync_eval.py --data_root $eval_path --filelist $filelist  --sync_root $sync_root --device $gpu
echo
echo "Evaluating visual quality...."
echo
python visual_quality.py --orig_root $gt_v_root --synt_root $eval_path --bbx_root $bbx_root --filelist $filelist

