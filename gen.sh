#!/bin/bash

python inf_test.py --filelist /dataDisk/mvlrs_v1/test.txt --video_root /dataDisk/mvlrs_v1/main --audio_root /dataDisk/mvlrs_v1/main --bbx_root /dataDisk/mvlrs_v1/bbx/main --save_root /home/wjd/Wav2Lip-master/camera_ready/v2_sma_lip_24_60k --ckpt_path /home/wjd/Wav2Lip-master/checkpoints/short/centre/checkpoint_step000060000.pth --avhubert_root /home/wjd/av_hubert/fairseq/examples --device 3 --ffmpeg ffmpeg

python inf_test.py --filelist /dataDisk/LRW_whole/test.txt --video_root /dataDisk/LRW_whole/LRW --audio_root /dataDisk/LRW_whole/bbx --bbx_root /dataDisk/LRW_whole/bbx --save_root /home/wjd/Wav2Lip-master/camera_ready/v2_sma_lip_24_60k_LRW --ckpt_path /home/wjd/Wav2Lip-master/checkpoints/short/centre/checkpoint_step000060000.pth --avhubert_root /home/wjd/av_hubert/fairseq/examples --device 3 --ffmpeg ffmpeg


