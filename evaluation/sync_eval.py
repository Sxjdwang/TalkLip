"""
This metric and code are adapted from 'wav2lip' GitHub repository
https://github.com/Rudrabha/Wav2Lip.
You may follow the above link for more information.

"""

#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess
import glob
import os
from tqdm import tqdm

def eval_sync(args):

    s = SyncNetInstance();

    s.loadParameters(args.initial_model);

    with open(args.filelist) as f:
        lines = f.readlines()

    all_videos = ['{}/{}.mp4'.format(args.data_root, line.strip().split()[0]) for line in lines[:12500]]

    print('path of one video to be evaluated is {}'.format(all_videos[-1]))

    prog_bar = tqdm(range(len(all_videos)))
    avg_confidence = 0.
    avg_min_distance = 0.

    args.tmp_dir += args.data_root.split('/')[-1]

    for videofile_idx in prog_bar:
        videofile = all_videos[videofile_idx]
        offset, confidence, min_distance = s.evaluate(args, videofile=videofile)
        avg_confidence += confidence
        avg_min_distance += min_distance
        prog_bar.set_description('Avg Confidence: {}, Avg Minimum Dist: {}'.format(round(avg_confidence / (videofile_idx + 1), 3), round(avg_min_distance / (videofile_idx + 1), 3)))
        prog_bar.refresh()

    print ('Average Confidence: {}'.format(avg_confidence/len(all_videos)))
    print ('Average Minimum Distance: {}'.format(avg_min_distance/len(all_videos)))

parser = argparse.ArgumentParser(description="SyncNet")

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='')
parser.add_argument('--batch_size', type=int, default='20', help='')
parser.add_argument('--vshift', type=int, default='15', help='')
parser.add_argument('--data_root', type=str, required=True, help='')
parser.add_argument('--filelist', type=str, required=True, help='')
parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='')
parser.add_argument('--reference', type=str, default="demo", help='')
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--sync_root', default='.', type=str)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
args.initial_model = '{}/{}'.format(args.sync_root, args.initial_model)

import sys
sys.path.append(args.sync_root)

from SyncNetInstance_calc_scores import *

eval_sync(args)









