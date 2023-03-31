#!/usr/bin/env bash

export PATH=/workspace/jiadong/interpVSR:$PATH

run.pl JOB=1:16 exp/decode.JOB.log \
   python upvideo.py --split train --rank JOB --nshard 16

python upvideo.py --split valid
python upvideo.py --split test
