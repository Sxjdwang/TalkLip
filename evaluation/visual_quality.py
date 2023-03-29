from skimage.metrics import structural_similarity
import numpy as np
import cv2
import argparse
from math import log10, sqrt


def readvideo(path):
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


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def main(args):

    with open(args.filelist) as f:
        lines = f.readlines()

    print("The folder of videos to be evaluated is {}".format(args.synt_root))
    ssim, psnr = [], []

    for line in lines:
        line = line.strip().split(' ')[0]
        gt = readvideo('{}/{}.mp4'.format(args.orig_root, line))
        synt = readvideo('{}/{}.mp4'.format(args.synt_root, line))
        bbxs = np.load('{}/{}.npy'.format(args.bbx_root, line))
        len_com = min(len(synt), len(gt))

        for i in range(len_com):
            bbx = bbxs[i]
            gt_re = cv2.resize(gt[i][bbx[1]:bbx[3], bbx[0]:bbx[2], :], (96, 96))
            synt_re = cv2.resize(synt[i][bbx[1]:bbx[3], bbx[0]:bbx[2], :], (96, 96))

            ssim.append(structural_similarity(gt_re, synt_re, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255))

            psnr.append(PSNR(gt_re, synt_re))

    ssim = np.array(ssim)
    psnr = np.array(psnr)

    print("PSNR and SSIM of generated videos are {} and {}".format(np.mean(psnr), np.mean(ssim)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate visual quality using PSNR and SSIM')

    parser.add_argument("--synt_root", help="Root folder of synthesized videos", required=True, type=str)
    parser.add_argument("--orig_root", help="Root folder of original videos", required=True, type=str)
    parser.add_argument('--bbx_root', help="Root folder of bounding boxes of faces", required=True, type=str)
    parser.add_argument('--filelist', help="Path of a file list containing all samples' name", required=True, type=str)

    args = parser.parse_args()

    main(args)


