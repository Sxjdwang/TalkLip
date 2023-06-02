"""
This file is adapted from 'wav2lip' GitHub repository
https://github.com/Rudrabha/Wav2Lip.

We design a new audio encoder with transformer
"""


import contextlib
import torch
from torch import nn
from torch.nn import functional as F
from .learn_sync import av_sync

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


class TalkLip(nn.Module):
    def __init__(self, audio_encoder, audio_num, res_layers=None):
        super(TalkLip, self).__init__()

        enc_channel = [6, 16, 32, 64, 128, 256, 512, 512]

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(enc_channel[0], enc_channel[1], kernel_size=7, stride=1, padding=3)),  # 16, 96, 96

            nn.Sequential(Conv2d(enc_channel[1], enc_channel[2], kernel_size=3, stride=2, padding=1),  # 32, 48, 48
                          Conv2d(enc_channel[2], enc_channel[2], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(enc_channel[2], enc_channel[2], kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(enc_channel[2], enc_channel[3], kernel_size=3, stride=2, padding=1),  # 64, 24,24
                          Conv2d(enc_channel[3], enc_channel[3], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(enc_channel[3], enc_channel[3], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(enc_channel[3], enc_channel[3], kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(enc_channel[3], enc_channel[4], kernel_size=3, stride=2, padding=1),  # 128, 12,12
                          Conv2d(enc_channel[4], enc_channel[4], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(enc_channel[4], enc_channel[4], kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(enc_channel[4], enc_channel[5], kernel_size=3, stride=2, padding=1),  # 256, 6,6
                          Conv2d(enc_channel[5], enc_channel[5], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(enc_channel[5], enc_channel[5], kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(enc_channel[5], enc_channel[6], kernel_size=3, stride=2, padding=1),  # 512, 3,3
                          Conv2d(enc_channel[6], enc_channel[6], kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2d(enc_channel[6], enc_channel[7], kernel_size=3, stride=1, padding=0),  # 1, 1
                          Conv2d(enc_channel[7], enc_channel[7], kernel_size=1, stride=1, padding=0)), ])

        self.audio_encoder = audio_encoder
        self.audio_map = nn.Linear(audio_num, enc_channel[-1])

        dec_channel = [512, 512, 512, 384, 256, 128, 64]
        upsamp_channel = []
        if res_layers is None:
            self.res_layers = len(dec_channel)
        else:
            self.res_layers = res_layers
        for i in range(len(dec_channel)):
            if i < self.res_layers:
                upsamp_channel.append(enc_channel[-i-1] + dec_channel[i])
            else:
                upsamp_channel.append(dec_channel[i])

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(enc_channel[-1], dec_channel[0], kernel_size=1, stride=1, padding=0), ),

            nn.Sequential(Conv2dTranspose(upsamp_channel[0], dec_channel[1], kernel_size=3, stride=1, padding=0),  # 3,3
                          Conv2d(dec_channel[1], dec_channel[1], kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2dTranspose(upsamp_channel[1], dec_channel[2], kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(dec_channel[2], dec_channel[2], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(dec_channel[2], dec_channel[2], kernel_size=3, stride=1, padding=1, residual=True), ),  # 6, 6

            nn.Sequential(Conv2dTranspose(upsamp_channel[2], dec_channel[3], kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(dec_channel[3], dec_channel[3], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(dec_channel[3], dec_channel[3], kernel_size=3, stride=1, padding=1, residual=True), ),  # 12, 12

            nn.Sequential(Conv2dTranspose(upsamp_channel[3], dec_channel[4], kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(dec_channel[4], dec_channel[4], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(dec_channel[4], dec_channel[4], kernel_size=3, stride=1, padding=1, residual=True), ),  # 24, 24

            nn.Sequential(Conv2dTranspose(upsamp_channel[4], dec_channel[5], kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(dec_channel[5], dec_channel[5], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(dec_channel[5], dec_channel[5], kernel_size=3, stride=1, padding=1, residual=True), ),  # 48, 48

            nn.Sequential(Conv2dTranspose(upsamp_channel[5], dec_channel[6], kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(dec_channel[6], dec_channel[6], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(dec_channel[6], dec_channel[6], kernel_size=3, stride=1, padding=1, residual=True), ), ])  # 96,96

        self.output_block = nn.Sequential(Conv2d(upsamp_channel[6], 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())
        self.ft = False
        self.sync_net = av_sync(audio_num, audio_num)

    def forward(self, sample, face_sequences, idAudio, B):

        input_dim_size = len(face_sequences.size())

        # input 1*F*T
        with torch.no_grad() if not self.ft else contextlib.ExitStack():
            enc_out = self.audio_encoder(**sample["net_input"])
        # T*B*C, B*T

        audio_embedding, audio_padding = enc_out['encoder_out'], enc_out['padding_mask']

        feats = []
        x = face_sequences
        # output is N*512*1*1
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)
        # T*B*C -> N*C -> N*512*1*1
        x = audio_embedding.permute(1, 0, 2).reshape(-1, audio_embedding.shape[2])[idAudio]
        x = self.audio_map(x).reshape(x.shape[0], 512, 1, 1)

        for i, f in enumerate(self.face_decoder_blocks):
            x = f(x)
            try:
                if i < self.res_layers:
                    x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs, audio_embedding#, ys_hat#, wer

    def get_aud_emb(self, sample):
        with torch.no_grad():
            enc_out = self.audio_encoder(**sample["net_input"])
        # T*B*C, B*T

        audio_embedding, audio_padding = enc_out['encoder_out'], enc_out['padding_mask']
        return audio_embedding


class TalkLip_disc_qual(nn.Module):
    def __init__(self):
        super(TalkLip_disc_qual, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)),  # 48,96

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),  # 48,48
                          nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 24,24
                          nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # 12,12
                          nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 6,6
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1), ),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                          nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def perceptual_forward(self, false_face_sequences):
        """
        force discriminator output given generated faces as input to 1
        Args:
            false_face_sequences: T*C*H*W

        Returns:

        """
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1),
                                                 torch.ones((len(false_feats), 1)).to(false_face_sequences.device)) #.cuda()

        return false_pred_loss

    def forward(self, face_sequences):

        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)
