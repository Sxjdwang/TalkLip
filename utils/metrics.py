"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
import numpy as np
import editdistance
from itertools import groupby


class asrMetrics(object):

    """
    A layer to add positional encodings to the inputs of a Transformer model.
    Formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, spaceIx):
        super(asrMetrics, self).__init__()
        self.spaceIx = spaceIx

    def convert_to_char_decoder(self, ys_hat, ys_pad):
        """Convert index to character.

        :param torch.Tensor seqs_hat: prediction (batch, seqlen)
        :param torch.Tensor seqs_true: reference (batch, seqlen)
        :return: token list of prediction
        :rtype list
        :return: token list of reference
        :rtype list
        """
        seqs_hat, seqs_true = [], []
        for i, y_hat in enumerate(ys_hat):
            y_true = ys_pad[i]
            eos_true = np.where(y_true == -1)[0]
            ymax = eos_true[0] if len(eos_true) > 0 else len(y_true)

            seq_hat = [idx.item() for idx in y_hat[:ymax]]
            seq_true = [idx.item() for idx in y_true if int(idx) != -1]
            seqs_hat.append(seq_hat)
            seqs_true.append(seq_true)
        return seqs_hat, seqs_true

    def compute_cer(self, predictionBatch, targetBatch):

        """
        Function to compute the Character Error Rate using the Predicted character indices and the Target character
        indices over a batch.
        CER is computed by dividing the total number of character edits (computed using the editdistance package)
        with the total number of characters (total => over all the samples in a batch).
        The <EOS> token at the end is excluded before computing the CER.

        predictionBatch: B*T,
        targetBatch: B*T, padding_value=-1
        """

        preds, trgts = self.convert_to_char(predictionBatch.cpu(), targetBatch.cpu())
        totalEdits = 0
        totalChars = 0

        for n in range(len(preds)):
            pred = preds[n]
            trgt = trgts[n]
            numEdits = editdistance.eval(pred, trgt)
            totalEdits = totalEdits + numEdits
            totalChars = totalChars + len(trgt)

        return totalEdits/totalChars

    def compute_wer(self, predictionBatch, targetBatch):

        """
        Function to compute the Word Error Rate using the Predicted character indices and the Target character
        indices over a batch. The words are obtained by splitting the output at spaces.
        WER is computed by dividing the total number of word edits (computed using the editdistance package)
        with the total number of words (total => over all the samples in a batch).
        The <EOS> token at the end is excluded before computing the WER. Words with only a space are removed as well.
        """

        preds, trgts = self.convert_to_char(predictionBatch.cpu(), targetBatch.cpu())
        totalEdits = 0
        totalWords = 0

        for n in range(len(preds)):
            pred = preds[n]
            trgt = trgts[n]

            predWords = np.split(pred, np.where(pred == self.spaceIx)[0])
            predWords = [predWords[0].tostring()] + [predWords[i][1:].tostring() for i in range(1, len(predWords)) if len(predWords[i][1:]) != 0]

            trgtWords = np.split(trgt, np.where(trgt == self.spaceIx)[0])
            trgtWords = [trgtWords[0].tostring()] + [trgtWords[i][1:].tostring() for i in range(1, len(trgtWords))]

            numEdits = editdistance.eval(predWords, trgtWords)
            totalEdits = totalEdits + numEdits
            totalWords = totalWords + len(trgtWords)

        return totalEdits/totalWords

    def convert_to_char(self, ys_hat, ys_pad, ilens=None):
        """Convert index to character.

        :param torch.Tensor seqs_hat: prediction (batch, seqlen)
        :param torch.Tensor seqs_true: reference (batch, seqlen)
        :return: token list of prediction
        :rtype list
        :return: token list of reference
        :rtype list
        """
        seqs_hat, seqs_true = [], []
        for i, y_hat in enumerate(ys_hat):
            y_true = ys_pad[i]

            seq_hat = np.array([idx[0] for idx in groupby(y_hat[:ilens[i]].cpu().detach().tolist())])
            seq_hat = seq_hat[seq_hat != 0]
            seq_true = [idx.item() for idx in y_true if int(idx) != -1]
            seqs_hat.append(seq_hat)
            seqs_true.append(seq_true)
        return seqs_hat, seqs_true

    def compute_both(self, predictionBatch, targetBatch):

        """
        Function to compute the Word Error Rate using the Predicted character indices and the Target character
        indices over a batch. The words are obtained by splitting the output at spaces.
        WER is computed by dividing the total number of word edits (computed using the editdistance package)
        with the total number of words (total => over all the samples in a batch).
        The <EOS> token at the end is excluded before computing the WER. Words with only a space are removed as well.
        """

        preds, trgts = self.convert_to_char_decoder(predictionBatch.cpu(), targetBatch.cpu())

        totalEdits = 0
        totalWords = 0

        totalEditsC = 0
        totalChars = 0

        for n in range(len(preds)):
            pred = np.array(preds[n])
            if len(pred) > 0:
                if pred[-1] == 39:
                    pred = pred[:-1]
            trgt = np.array(trgts[n])

            numEdits = editdistance.eval(pred, trgt)
            totalEditsC = totalEditsC + numEdits
            totalChars = totalChars + len(trgt)

            predWords = np.split(pred, np.where(pred == self.spaceIx)[0])
            predWords = [predWords[0].tostring()] + [predWords[i][1:].tostring() for i in range(1, len(predWords)) if len(predWords[i][1:]) != 0]

            trgtWords = np.split(trgt, np.where(trgt == self.spaceIx)[0])
            trgtWords = [trgtWords[0].tostring()] + [trgtWords[i][1:].tostring() for i in range(1, len(trgtWords))]

            numEdits = editdistance.eval(predWords, trgtWords)
            totalEdits = totalEdits + numEdits
            totalWords = totalWords + len(trgtWords)

        return totalEditsC/totalChars, totalEdits/totalWords
