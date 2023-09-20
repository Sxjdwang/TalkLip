#!/bin/bash

eval_path=$1
filelist=$2
sync_root=$3
gt_v_root=$4
bbx_root=$5
ckpt_wer2=$6
ckpt_wer1=$9
data_root_wer2=$7
avhubert_root=$8
gpu=${10:-0}

sh vq_sync_eval.sh $eval_path $filelist $sync_root $gt_v_root $bbx_root $gpu

echo
echo "Evaluating reading intelligibility with the teacher force mode...."
echo
CUDA_VISIBLE_DEVICES=$gpu python teacher_force_wer.py --data_root $data_root_wer2 --video_root $eval_path --ckpt_path $ckpt_wer2

echo
echo "Evaluating reading intelligibility with AV-Hubert...."
echo
subset=$(basename $eval_path)
python toavhform.py --video_root $eval_path --audio_root / --subset $subset
cp ../datalist/test.wrd ../datalist/$subset.wrd

CUDA_VISIBLE_DEVICES=$gpu python $avhubert_root/infer_s2s.py --config-dir $avhubert_root/conf/ --config-name s2s_decode.yaml dataset.gen_subset=$subset common_eval.path=$ckpt_wer1 \
common_eval.results_path=$subset override.modalities=['video'] common.user_dir=$avhubert_root \
+override.data=/workspace2/jiadong/TalkLip/datalist +override.label_dir=/workspace2/jiadong/TalkLip/datalist > $subset.txt

tail -6 $subset.txt
rm $subset.txt

if [ "$subset" != "test" ]; then
    rm ../datalist/$subset.wrd
    rm ../datalist/$subset.tsv
fi


