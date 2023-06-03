#!/usr/bin/env bash

export PATH=preparation:$PATH

gpu=2
video_root=/workspace2/jiadong/mvlrs_v1/main
bbx_root=/workspace2/jiadong/mvlrs_v1/bbx/main
file_list_dir=/workspace2/jiadong/mvlrs_v1
num_thread=6

# extract bounding boxes from train set
run.pl JOB=1:$num_thread exp/decode.JOB.log \
   python preparation/bbx_extract.py --rank JOB --nshard $num_thread --filelist $file_list_dir/train.txt --video_root $video_root --bbx_root $bbx_root --gpu $gpu

# extract bounding boxes from valid set
python preparation/bbx_extract.py --rank 1 --nshard 1 --filelist $file_list_dir/valid.txt --video_root $video_root --bbx_root $bbx_root --gpu $gpu
# extract bounding boxes from test set
python preparation/bbx_extract.py --rank 1 --nshard 1 --filelist $file_list_dir/test.txt --video_root $video_root --bbx_root $bbx_root --gpu $gpu

