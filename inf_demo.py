import torch
import platform
import math
import numpy as np
import os, cv2, argparse, subprocess

from tqdm import tqdm
from torch.nn import functional as F
from argparse import Namespace
from python_speech_features import logfbank
from fairseq import checkpoint_utils, utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, populate_dataclass, merge_with_parent
from scipy.io import wavfile
from utils.data_avhubert import collater_audio, emb_roi2im

from models.talklip import TalkLip
import face_detection


def build_encoder(hubert_root, path='config.yaml'):

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)

    import sys
    sys.path.append(hubert_root)
    from avhubert.hubert_asr import HubertEncoderWrapper, AVHubertSeq2SeqConfig

    # cfg = merge_with_parent(AVHubertSeq2SeqConfig(), cfg)
    arg_overrides = {
        "dropout": cfg.dropout,
        "activation_dropout": cfg.activation_dropout,
        "dropout_input": cfg.dropout_input,
        "attention_dropout": cfg.attention_dropout,
        "mask_length": cfg.mask_length,
        "mask_prob": cfg.mask_prob,
        "mask_selection": cfg.mask_selection,
        "mask_other": cfg.mask_other,
        "no_mask_overlap": cfg.no_mask_overlap,
        "mask_channel_length": cfg.mask_channel_length,
        "mask_channel_prob": cfg.mask_channel_prob,
        "mask_channel_selection": cfg.mask_channel_selection,
        "mask_channel_other": cfg.mask_channel_other,
        "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
        "encoder_layerdrop": cfg.layerdrop,
        "feature_grad_mult": cfg.feature_grad_mult,
    }
    if cfg.w2v_args is None:
        state = checkpoint_utils.load_checkpoint_to_cpu(
            cfg.w2v_path, arg_overrides
        )
        w2v_args = state.get("cfg", None)
        if w2v_args is None:
            w2v_args = convert_namespace_to_omegaconf(state["args"])
        cfg.w2v_args = w2v_args
    else:
        state = None
        w2v_args = cfg.w2v_args
        if isinstance(w2v_args, Namespace):
            cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                w2v_args
            )

    w2v_args.task.data = cfg.data
    task_pretrain = tasks.setup_task(w2v_args.task)

    task_pretrain.load_state_dict(torch.load('/home/wjd/Talklip/task_state.pt'))

    encoder_ = task_pretrain.build_model(w2v_args.model)
    encoder = HubertEncoderWrapper(encoder_)
    if state is not None and not cfg.no_pretrained_weights:
        # set strict=False because we omit some modules
        del state['model']['mask_emb']
        encoder.w2v_model.load_state_dict(state["model"], strict=False)

    encoder.w2v_model.remove_pretraining_modules()
    return encoder, encoder.w2v_model.encoder_embed_dim


def parse_filelist(file_list, save_root, check):

    with open(file_list) as f:
        lines = f.readlines()

    if check:
        sample_paths = []
        for line in lines:
            line = line.strip().split()[0]
            if not os.path.exists('{}/{}.mp4'.format(save_root, line)):
                sample_paths.append(line)
    else:
        sample_paths = [line.strip().split()[0] for line in lines]

    return sample_paths


class Talklipdata(object):

    def __init__(self, args):
        self.data_root = args.video_root
        self.bbx_root = args.bbx_root
        self.audio_root = args.audio_root
        self.samples = parse_filelist(args.filelist, args.save_root, args.check)
        self.stack_order_audio = 4
        self.crop_size = 96

    def prepare_window(self, window):
        # T x 3 x H x W
        x = window / 255.
        x = x.permute((0, 3, 1, 2))

        return x

    def croppatch(self, images, bbxs):
        patch = np.zeros((images.shape[0], self.crop_size, self.crop_size, 3))
        width = images.shape[1]
        for i, bbx in enumerate(bbxs):
            bbx[2] = min(bbx[2], width)
            bbx[3] = min(bbx[3], width)
            patch[i] = cv2.resize(images[i, bbx[1]:bbx[3], bbx[0]:bbx[2], :], (self.crop_size, self.crop_size))
        return patch

    def audio_visual_align(self, audio_feats, video_feats):
        diff = len(audio_feats) - len(video_feats)
        if diff < 0:
            audio_feats = np.concatenate(
                [audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
        elif diff > 0:
            left = diff // 2
            right = diff - left
            audio_feats = audio_feats[left:-right]
            # audio_feats = audio_feats[:-diff]
        return audio_feats

    def fre_audio(self, wav_data, sample_rate):
        def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames, 4 frames of tf forms a new frame of tf
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats

        audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)  # [T, F]
        audio_feats = stacker(audio_feats, self.stack_order_audio)  # [T/stack_order_audio, F*stack_order_audio]
        return audio_feats

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        imgs = []
        while True:
            ret, frame = cap.read()
            if ret:
                imgs.append(frame)
            else:
                break
        cap.release()
        return imgs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """

        Args:
            idx:

        Returns:
            x (N*6*96*96): concatenation of N identity images (different) and mask images (same)
            spectrogram (N_a * 321): spectrogram of whole wav
            idAudio ((N*7)): matched audio index
            y (N*3*96*96): ground truth images
        """
        sample = self.samples[idx]

        video_path = '{}/{}.mp4'.format(self.data_root, sample)
        bbx_path = '{}/{}.npy'.format(self.bbx_root, sample)
        wav_path = '{}/{}.wav'.format(self.audio_root, sample)

        bbxs = np.load(bbx_path)

        imgs = np.array(self.load_video(video_path))
        volume = len(imgs)

        sampRate, wav = wavfile.read(wav_path)
        spectrogram = self.fre_audio(wav, sampRate)
        spectrogram = torch.tensor(spectrogram) # T'* F
        with torch.no_grad():
            spectrogram = F.layer_norm(spectrogram, spectrogram.shape[1:])

        pickedimg = list(range(volume))
        poseImgRaw = np.array(pickedimg)
        poseImg = self.croppatch(imgs[poseImgRaw], bbxs[poseImgRaw])
        idImgRaw = np.zeros(volume, dtype=np.int32)
        idImg = self.croppatch(imgs[idImgRaw], bbxs[idImgRaw])

        poseImg = torch.tensor(poseImg, dtype=torch.float32)  # T*3*96*96
        idImg = torch.tensor(idImg, dtype=torch.float32)  # T*3*96*96

        spectrogram = self.audio_visual_align(spectrogram, imgs)

        pose_inp = self.prepare_window(poseImg)
        gt = pose_inp.clone()
        # mask off the bottom half
        pose_inp[:, :, pose_inp.shape[2] // 2:] = 0.

        id_inp = self.prepare_window(idImg)
        inp = torch.cat([pose_inp, id_inp], dim=1)

        pickedimg, bbxs = torch.tensor(pickedimg), torch.tensor(bbxs)

        imgs = torch.from_numpy(imgs)

        return inp, spectrogram, gt, volume, pickedimg, imgs, bbxs, sample


def collate_fn(dataBatch):
    """

    Args:
        dataBatch:

    Returns:
        xBatch: input T_sum*6*96*96, concatenation of all video chips in the time dimension
        yBatch: output T_sum*3*96*96
        inputLenBatch: bs
        inputLenRequire: bs
        audioBatch: bs*T'*321 or T_sum*1*80*16
        idAudio: (bs*N*7)
        targetBatch: bs*L*1
        videoBatch: bs*T''*3*96*96
        pickedimg: (bs*N*5)
        videoBatch: bs*T''*3*96*96
    """

    xBatch = torch.cat([data[0] for data in dataBatch], dim=0)
    yBatch = torch.cat([data[2] for data in dataBatch], dim=0)
    inputLenBatch = [data[3] for data in dataBatch]

    audioBatch, padding_mask = collater_audio([data[1] for data in dataBatch], max(inputLenBatch))

    audiolen = audioBatch.shape[2]
    idAudio = torch.cat([data[4] + audiolen * i for i, data in enumerate(dataBatch)], dim=0)

    pickedimg = [data[4] for data in dataBatch]
    videoBatch = [data[5] for data in dataBatch]
    bbxs = [data[6] for data in dataBatch]
    names = [data[7] for data in dataBatch]

    return xBatch, audioBatch, idAudio, yBatch, padding_mask, pickedimg, videoBatch, bbxs, names


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def prepare_window(window):
    # T x 3 x H x W
    x = window / 255.
    x = x.permute((0, 3, 1, 2))

    return x


def detect_bbx(frames, fa):

    height, width, _ = frames[0].shape
    batches = [frames[i:i + 32] for i in range(0, len(frames), 32)]

    bbxs = list()
    for fb in batches:
        preds = fa.get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            if f is None:
                htmp = int((height - 96) / 2)
                wtmp = int((width - 96) / 2)
                x1, y1, x2, y2 = wtmp, htmp, wtmp + 96, htmp + 96
            else:
                x1, y1, x2, y2 = f
            bbxs.append([x1, y1, x2, y2])
    bbxs = np.array(bbxs)
    return bbxs


def croppatch(images, bbxs, crop_size=96):
    patch = np.zeros((images.shape[0], crop_size, crop_size, 3))
    width = images.shape[1]
    for i, bbx in enumerate(bbxs):
        bbx[2] = min(bbx[2], width)
        bbx[3] = min(bbx[3], width)
        patch[i] = cv2.resize(images[i, bbx[1]:bbx[3], bbx[0]:bbx[2], :], (crop_size, crop_size))
    return patch


def audio_visual_pad(audio_feats, video_feats):
    diff = len(audio_feats) - len(video_feats)
    repeat = 1
    if diff > 0:
        repeat = math.ceil(len(audio_feats) / len(video_feats))
        video_feats = torch.repeat_interleave(video_feats, repeat, dim=0)

    diff = len(audio_feats) - len(video_feats)
    video_feats = video_feats[:diff]

    return video_feats, repeat, diff


def fre_audio(wav_data, sample_rate):
    def stacker(feats, stack_order):
        """
        Concatenating consecutive audio frames, 4 frames of tf forms a new frame of tf
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
        return feats

    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)  # [T, F]
    audio_feats = stacker(audio_feats, 4)  # [T/stack_order_audio, F*stack_order_audio]
    return audio_feats


def load_video(path):
    cap = cv2.VideoCapture(path)
    imgs = []
    while True:
        ret, frame = cap.read()
        if ret:
            imgs.append(frame)
        else:
            break
    cap.release()
    return imgs


def data_preprocess(args, face_detector):
    video_path = args.video_path
    wav_path = args.wav_path

    imgs = np.array(load_video(video_path))

    sampRate, wav = wavfile.read(wav_path)
    spectrogram = fre_audio(wav, sampRate)
    spectrogram = torch.tensor(spectrogram)  # T'* F
    with torch.no_grad():
        spectrogram = F.layer_norm(spectrogram, spectrogram.shape[1:])

    bbxs = detect_bbx(imgs, face_detector)

    poseImg = croppatch(imgs, bbxs)

    poseImg = torch.tensor(poseImg, dtype=torch.float32)  # T*3*96*96

    poseImg, repeat, diff = audio_visual_pad(spectrogram, poseImg)

    if repeat > 1:
        imgs = np.repeat(imgs, repeat, axis=0)
        imgs = imgs[:diff]
        bbxs = np.repeat(bbxs, repeat, axis=0)
        bbxs = bbxs[:diff]

    pose_inp = prepare_window(poseImg)
    id_inp = pose_inp.clone()
    # mask off the bottom half
    pose_inp[:, :, pose_inp.shape[2] // 2:] = 0.

    inp = torch.cat([pose_inp, id_inp], dim=1)

    bbxs = torch.tensor(bbxs)
    imgs = torch.from_numpy(imgs)

    audioBatch = spectrogram.unsqueeze(dim=0).transpose(1, 2)
    padding_mask = (torch.BoolTensor(1, inp.shape[0]).fill_(False))
    idAudio = torch.arange(inp.shape[0])

    return inp, audioBatch, idAudio, padding_mask, [imgs], [bbxs]


def synt_demo(face_detector, device, model, args):

    model.eval()

    inps, spectrogram, idAudio, padding_mask, imgs, bbxs = data_preprocess(args, face_detector)

    inps = inps.to(device)
    spectrogram = spectrogram.to(device)
    padding_mask = padding_mask.to(device)

    sample = {'net_input': {'source': {'audio': spectrogram, 'video': None}, 'padding_mask': padding_mask, 'prev_output_tokens': None},
              'target_lengths': None, 'ntokens': None, 'target': None}

    prediction, _ = model(sample, inps, idAudio, spectrogram.shape[0])

    file_size = imgs[0].shape[1]
    processed_img = emb_roi2im([idAudio], imgs, bbxs, prediction, device)

    out_path = '{}.mp4'.format(args.save_path)
    tmpvideo = '{}.avi'.format(args.save_path)

    out = cv2.VideoWriter(tmpvideo, cv2.VideoWriter_fourcc(*'DIVX'), 25, (file_size, file_size))

    for j, im in enumerate(processed_img[0]):
        im = im.cpu().clone().detach().numpy().astype(np.uint8)
        out.write(im)

    out.release()

    command = '{} -y -i {} -i {} -strict -2 -q:v 1 {} -loglevel quiet'.format(args.ffmpeg, args.wav_path, tmpvideo, out_path) #

    subprocess.call(command, shell=platform.system() != 'Windows')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Synthesize a video conditioned on a different audio')

    parser.add_argument("--video_path", help="Root folder of video", required=True, type=str)
    parser.add_argument("--wav_path", help="Root folder of audio", required=True, type=str)
    parser.add_argument("--save_path", help="a directory to save the synthesized video", default='tmp', type=str)
    parser.add_argument('--ckpt_path', help='pretrained checkpoint', required=True, type=str)
    parser.add_argument('--avhubert_root', help='Path of av_hubert root', required=True, type=str)
    parser.add_argument('--check', help='whether filter out videos which have been synthesized in save_root', default=False, type=bool)
    parser.add_argument('--ffmpeg', default='ffmpeg', type=str)
    parser.add_argument('--device', default=0, type=int)

    args = parser.parse_args()

    device = "cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu"

    model = TalkLip(*build_encoder(args.avhubert_root)).to(device)

    fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                      device=device)

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device)["state_dict"])
    with torch.no_grad():
        synt_demo(fa, device, model, args)
