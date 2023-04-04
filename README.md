# TalkLip net

This repo is the official implementation of 'Seeing What You Said: Talking Face Generation Guided by a Lip Reading Expert', CVPR 2023.

[Paper](http://arxiv.org/abs/2303.17480)

## Prerequisite 

1. `pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html`
2. Install [AV-Hubert](https://github.com/facebookresearch/av_hubert)
3. Install supplementary packages via `pip install -r requirements.txt`
5. Install ffmpeg. We adopt version=4.3.2. Please double check wavforms extracted from mp4 files. Extracted wavforms should not contain prefix of 0. If you use anaconda, you can refer to `conda install -c conda-forge ffmpeg==4.2.3`
6. Download the pre-trained checkpoint of face detector [pre-trained model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) and put it to `face_detection/detection/sfd/s3fd.pth`. Alternative [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/prajwal_k_research_iiit_ac_in/EZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg?e=qTasa8).


## Dataset and pre-processing

1. Download [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) for training and evaluation. Note that we do not use the pretrain set.
2. Download [LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) for evaluation.
3. To extract wavforms from mp4 files:
```
python preparation/audio_extract.py --filelist $path1  --video_root $path2 --audio_root $path3
```
- $path1: a txt file containing names of videos. We provide the filelist of LRW test set as an example in the datalist directory.  
- $path2: root directory of videos. For example: $LRS2_root/main
- $path3: root directory for saving wavforms
- other optional arguments: please refer to audio_extract.py  

4. To detect bounding boxes in videos and save it:
```
python preparation/bbx_extract.py --filelist #path1  --video_root $path2 --bbx_root $path4
```
- $path4: a root directory for saving detected bounding boxes

## Train 
We are aranging codes and will release them before June

## Test 
The below command is to synthesize videos for quantitative evaluation in our paper.
```
python inf_test.py --filelist $path1 --video_root $path2 --audio_root $path3 \
--bbx_root $path4 --save_root $path5 --ckpt_path $path6 --avhubert_root $avhubert_root
```
- $path5: root directory for saving synthesized videos
- $path6: a pre-trained checkpoint of TalkLip net
- $avhubert_root: path of root of avhubert (should like xxx/av_hubert/avhubert)

## Demo
I update the inf_demo.py on 4/April as I previously suppose that the height and width of output videos are the same when I set cv2.VideoWriter().

If you want to reenact the lip movement of a video with a different speech, you can use the following command. 
```
python inf_demo.py --video_path $path8 --wav_path $path9 --ckpt_path $path6 --avhubert_root $avhubert_root
```
- $path8: a video file (end with .mp4)
- $path9: a audio file (end with .wav)

## Evaluation

Please follow README.md in the evaluation directory


Checkpoints
----------
| Model  | Description |  Link  | 
| :-------------: | :---------------: | :---------------: |
| TalkLip (g)  | TalkLip net with the global audio encoder | [Link](https://drive.google.com/file/d/1iBXJmkS8rjzTBE6XOC3-XiXufEK2f1dj/view?usp=share_link)  |
| TalkLip (g+c)  | TalkLip net with the global audio encoder and contrastive learning | [Link](https://drive.google.com/file/d/1nfPHicsHr2bOzvkdyoMk_GCYzJ3fqvI-/view?usp=share_link) |
| Lip reading observer 1 | AV-hubert (large) fine-tuned on LRS2 | [Link](https://drive.google.com/file/d/1wOsiXKLOeScrU6XuzebYA6Y-9ncd8-le/view?usp=share_link) |
| Lip reading observer 2 | Conformer lip-reading network | [Link](https://drive.google.com/file/d/16tpyaXLLTYUnIBT_YEWQ5ui6xUkBGcpM/view?usp=share_link) |
| Lip reading expert | lip-reading network for training of talking face generation | [Link](https://drive.google.com/file/d/1sKCBak-odjUnvEJ99us7gJlcXQSopdmu/view?usp=share_link) |


