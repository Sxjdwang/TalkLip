"""
Author: Jiadong Wang

Note: class SupConLoss is adapted from 'SupContrast' GitHub repository
https://github.com/HobbitLong/SupContrast
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, anchor_audio, positive_video, negative_audio):
        """

        Args:
            anchor_audio: B * C
            positive_video: B * C
            negative_audio: N * B * C

        Returns:

        """
        device = anchor_audio.device

        positive_video = positive_video.unsqueeze(0)
        contrast_feature = torch.cat([positive_video, negative_audio], dim=0)

        # compute logits
        anchor_dot_contrast = torch.sum(anchor_audio.unsqueeze(0)*contrast_feature, dim=-1) / self.temperature
        anchor_dot_contrast = anchor_dot_contrast.permute(1, 0)

        mask = torch.zeros_like(anchor_dot_contrast).to(device)
        mask[:, 0] = 1.
        mask = mask.detach()

        loss = self.lossBySimi(anchor_dot_contrast, mask)

        return loss

    def lossBySimi(self, similarity, mask):
        # for numerical stability
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        sum_log_prob_pos = (mask * log_prob).sum(1)
        mean_log_prob_pos = sum_log_prob_pos

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss


class av_sync(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(av_sync, self).__init__()

        self.Linearaudio = nn.Linear(in_channel, out_channel)

        self.criterion = SupConLoss()

        self.dropout = nn.Dropout(p=0.25)

        self.maxproportion = 60

    def preprocess_local(self, video, audio, pickedimg):
        video_cl = []
        for i, singlepick in enumerate(pickedimg):
            idimg = pickedimg[i]
            for j in range(len(idimg)):
                video_cl.append(video[idimg[j], i])
        return torch.stack(video_cl, dim=0), audio.view(-1, 512)

    def forward(self, video, audio, pickedimg=None):
        """

        :param video: tensor-> batch*ndim
        :param audio: tensor-> batch*ndim
        :return:
        """
        if pickedimg is not None:
            video, audio = self.preprocess_local(video, audio, pickedimg)

        video_cl_norm = F.normalize(video, dim=1)

        audio = F.normalize(audio, dim=1)
        audio_cl = self.Linearaudio(audio)
        audio_cl_norm = F.normalize(audio_cl, dim=1)

        n_negatives = min(100, int(video_cl_norm.shape[0]/2))
        # generate a audio_cl_norm.shape[0] * n_negatives matrix containing indices.
        neg_idxs = torch.multinomial(torch.ones((audio_cl_norm.shape[0], audio_cl_norm.shape[0] - 1), dtype=torch.float), n_negatives)
        tszs = torch.tensor(list(range(audio_cl_norm.shape[0]))).view(-1, 1)
        neg_idxs[neg_idxs >= tszs] += 1

        negs_a = audio_cl_norm[neg_idxs.view(-1)]
        negs_a = negs_a.view(
            audio_cl_norm.shape[0], n_negatives, audio_cl_norm.shape[1]
        ).permute(1, 0, 2)  # to NxBxC

        dist_loss = self.criterion(audio_cl_norm, video_cl_norm, negs_a)

        return dist_loss

