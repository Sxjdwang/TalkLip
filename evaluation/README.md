# Evaluateion of PSNR, SSIM, LSE-C, LSE-D, ACC, WER1, WER2.

# Evaluating all metrics on LRS2 via one bash script

Before using this script, please read the below instruction of each metric first, as there are some preparations need to be done.
```
sh eval_lrs2.sh $synt_root $filelist $syncnet_python_root $orig_root $bbx_root $ckpt_path_conformer $data_root $avhubert_root \
$ckpt_path_av_hubert $gpu
```
- synt_root: the directory containing synthesized videos, which is the $syn_video_root when you run inf_test.py. 
- filelist: refers to $filelist when you run inf_test.py, e.g. xxx/mvlrs_v1/test.txt.
- syncnet_python_root: the root of syncnet_python
- orig_root: the root of original data, e.g. xxx/mvlrs_v1/main.
- bbx_root: refers to $bbx_root in 'Dataset and pre-processing'
- ckpt_path_conformer: the path where you save 'Lip reading observer 2'
- data_root: a directory containing $orig_root and $filelist, e.g. xxx/mvlrs_v1.
- avhubert_root: the root of AV-hubert, e.g. xxx/av_hubert/avhubert.
- ckpt_path_av_hubert: the path where you save 'Lip reading observer 1'
- gpu: the index of gpu to conduct evaluation.

## Visual quality

In our paper, we adopt metrics of PSNR and SSIM. You can get these results by the following commands:
```
python visual_quality.py --orig_root $orig_root --synt_root $synt_root --bbx_root $bbx_root --filelist $filelist
```

## lip-speech synchronization

1. Clone the SyncNet repository

```
git clone https://github.com/joonson/syncnet_python.git 
```

2. download pre-trained model

```
cd $syncnet_python_root
sh download_model.sh
```

3. copy a python file to the root of syncnet_python 
```
cp SyncNetInstance_calc_scores.py $syncnet_python_root
```

4. evaluate synthesized videos
```
python sync_eval.py --data_root $path5 --filelist $path1 --sync_root $syncnet_python_root
```
- path5 is the path of the syncnet_python 

## AV-Hubert WER evaluation 

1. Compose a filelist file compatible with AV-Hubert:
```
python toavhform.py --video_root $path5 --audio_root $path10
```
- $path10: you can put any path since audio is not evolved in lip-reading evaluation

2. download Lip reading observer 1 [link](https://drive.google.com/file/d/1wOsiXKLOeScrU6XuzebYA6Y-9ncd8-le/view?usp=share_link)

3. Evaluate Word Error Rate 
```
cd $avhubert_root
python infer_s2s.py --config-dir ./conf/ --config-name s2s_decode.yaml dataset.gen_subset=test common_eval.path=$path11 \
common_eval.results_path=decode/s2s/ override.modalities=['video'] \
common.user_dir=$avhubert_root +override.data=$TalkLip_root/datalist +override.label_dir=$TalkLip_root/datalist
```
$path11: where you save large_ft_lrs2.pt

## Conformer WER evaluation

1. Download Lip reading observer 2 [link](https://drive.google.com/file/d/16tpyaXLLTYUnIBT_YEWQ5ui6xUkBGcpM/view?usp=share_link)

2. Evaluate Word Error Rate 
```
python teacher_force_wer.py --data_root $path12 --video_root $path5 \
--ckpt_path $path13
```
- $path12: a directory contain $path1 and $path2
- $path13: where you save conformer.pt

## Accuracy evaluation on LRW

We adopt [Multi-head Visual-audio Memory](https://github.com/ms-dot-k/Multi-head-Visual-Audio-Memory) to evaluate classification accuracy of synthesized videos on LRW.

As their environment is different with ours, we recommend to create a separable environment following their requirement [Multi-head Visual-audio Memory](https://github.com/ms-dot-k/Multi-head-Visual-Audio-Memory).

Afterwards, you can evaluate synthesized videos by the following commands.

```
cd $Multi-head-Visual-audio-Memory_root
python test.py --lrw $path5 --checkpoint $Multi-head-Visual-Audio-Memory_root/Pretrained_Ckpt.ckpt --batch_size 40 --radius 16 --slot 112 --head 8 --test_aug --gpu 0
```


