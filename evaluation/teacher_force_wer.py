"""
Author: Jiadong Wang

Note: class LRS2Main is adapted from 'deep_avsr' GitHub repository
https://github.com/lordmartian/deep_avsr
"""

import torch
import argparse
import numpy as np
import cv2

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import os, sys
sys.path.append(os.getcwd().replace('/evaluation', ''))
from models.conformer_lip_reading import E2E

char_dict = {" ":1, "'":22, "1":30, "0":29, "3":37, "2":32, "5":34, "4":38, "7":36, "6":35, "9":31, "8":33,
             "A":5, "C":17, "B":20, "E":2, "D":12, "G":16, "F":19, "I":6, "H":9, "K":24, "J":25, "M":18,
             "L":11, "O":4, "N":7, "Q":27, "P":21, "S":8, "R":10, "U":13, "T":3, "W":15, "V":23, "Y":14,
             "X":26, "Z":28, "<EOS>":39}    #character to index mapping

index_dict = {1:" ", 22:"'", 30:"1", 29:"0", 37:"3", 32:"2", 34:"5", 38:"4", 36:"7", 35:"6", 31:"9", 33:"8",
             5:"A", 17:"C", 20:"B", 2:"E", 12:"D", 16:"G", 19:"F", 6:"I", 9:"H", 24:"K", 25:"J", 18:"M",
             11:"L", 4:"O", 7:"N", 27:"Q", 21:"P", 8:"S", 10:"R", 13:"U", 3:"T", 15:"W", 23:"V", 14:"Y",
             26:"X", 28:"Z", 39:"<EOS>"}    #index to character reverse mapping


class LRS2Main(Dataset):

    """
    A custom dataset class for the LRS2 main (includes train, val, test) dataset
    """

    def __init__(self, datadir, imagedir, charToIx, im_std, im_mean):
        super(LRS2Main, self).__init__()

        with open("{}/test.txt".format(datadir), "r") as f:
            lines = f.readlines()

        self.txtlist = ["{}/main/{}.txt".format(datadir, line.strip().split(' ')[0]) for line in lines]
        self.videolist = ["{}/{}.mp4".format(imagedir, line.strip().split(' ')[0]) for line in lines]

        self.charToIx = charToIx

        self.im_std = im_std
        self.im_mean = im_mean

    def req_input_length(self, trgt):
        """
        Function to calculate the minimum required input length from the target.
        Req. Input Length = No. of unique chars in target + No. of repeats in repeated chars (excluding the first one)
        """
        reqLen = len(trgt)
        lastChar = trgt[0]
        for i in range(1, len(trgt)):
            if trgt[i] != lastChar:
                lastChar = trgt[i]
            else:
                reqLen = reqLen + 1
        return reqLen

    def collate_fn(self, dataBatch):
        """
        Collate function definition used in Dataloaders.
        """
        inputBatch = pad_sequence([data[0] for data in dataBatch], batch_first=True)
        if not any(data[1] is None for data in dataBatch):
            # targetBatch = torch.cat([data[1] for data in dataBatch])
            targetBatch = pad_sequence([data[1].unsqueeze(dim=1) for data in dataBatch], batch_first=True,
                                       padding_value=-1)
        else:
            targetBatch = None

        inputLenBatch = torch.stack([data[2] for data in dataBatch])
        inputLenRequire = torch.stack([data[3] for data in dataBatch])

        return inputBatch, targetBatch, inputLenBatch, inputLenRequire

    def __getitem__(self, index):

        videoPath = self.videolist[index]
        targetPath = self.txtlist[index]

        with open(targetPath, "r") as f:
            trgt = f.readline().strip()[7:]

        trgt = [self.charToIx[char] for char in trgt]
        trgt = torch.tensor(trgt)

        captureObj = cv2.VideoCapture(videoPath)
        roiSequence = list()
        roiSize = 112

        while (captureObj.isOpened()):
            ret, frame = captureObj.read()
            if ret == True:
                grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                grayed = cv2.resize(grayed, (224, 224))
                roi = grayed[int(112 - (roiSize / 2)):int(112 + (roiSize / 2)),
                      int(112 - (roiSize / 2)):int(112 + (roiSize / 2))]
                roiSequence.append(roi)
            else:
                break
        captureObj.release()
        inp = np.stack(roiSequence)

        inpLen = len(inp)
        reqInpLen = self.req_input_length(trgt)
        reqInpLen = max(reqInpLen, inpLen)

        inp = (torch.from_numpy(inp) / 255. - self.im_mean) / self.im_std
        inpLen = torch.tensor(inpLen)
        reqInpLen = torch.tensor(reqInpLen)

        return inp, trgt, inpLen, reqInpLen


    def __len__(self):
        return len(self.txtlist)


def main(args):

    device = "cuda:{}".format(args.device)  if torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #declaring the train and validation datasets and their corresponding dataloaders
    valData = LRS2Main(args.data_root, args.video_root, char_dict, args.video_std, args.video_mean)

    valLoader = DataLoader(valData, batch_size=1, collate_fn=valData.collate_fn, shuffle=False, num_workers=args.num_workers)

    #declaring the model, optimizer, scheduler and the loss function

    model = E2E(512, 40, 'conformer_arg.txt', char_dict[" "])

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.to(device)

    evalCER, evalWER = 0, 0

    for batch, (inputBatch, targetBatch, inputLenBatch, inputLenRequire) in enumerate(tqdm(valLoader, leave=False, desc="Eval", ncols=75)):

        inputBatch, targetBatch = (inputBatch.float()).to(device, non_blocking=True, dtype=torch.float32), (targetBatch.long()).to(device, non_blocking=True)
        inputLenBatch, inputLenRequire = (inputLenBatch.int()).to(device, non_blocking=True), (inputLenRequire.int()).to(device, non_blocking=True)

        model.eval()
        with torch.no_grad():
            cer, wer = model(inputBatch, inputLenBatch, targetBatch.squeeze(dim=2), inputLenRequire)

        reduce_cer = cer * inputBatch.shape[0]
        reduce_wer = wer * inputBatch.shape[0]

        evalCER = evalCER + reduce_cer
        evalWER = evalWER + reduce_wer

    print("Val.CER: %.3f  Val.WER: %.3f" % (evalCER/len(valLoader.dataset), evalWER/len(valLoader.dataset)))

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate video intelligibility via a Conformer model with the teacher-force mode')

    parser.add_argument("--data_root", help="Root folder of the LRS2 dataset, containing test.txt, folders of main and pretrain", required=True, type=str)
    parser.add_argument("--video_root", help="Root folder of synthesized videos ", required=True, type=str)
    parser.add_argument('--ckpt_path', help='path of the trained lip-reading model', required=True, type=str)

    parser.add_argument('--video_std', help='normalization std', default=0.1688, type=float)
    parser.add_argument('--video_mean', help='normalization mean', default=0.4161, type=float)

    parser.add_argument('--seed', default=19220297, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    args = parser.parse_args()

    main(args)
