from os import path

import numpy as np
import argparse, os, cv2
from tqdm import tqdm
import math

import face_detection

def process_video_file(samplename, args, fa):
    vfile = '{}/{}.mp4'.format(args.video_root, samplename)
    video_stream = cv2.VideoCapture(vfile)

    frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)
    height, width, _ = frames[0].shape

    fulldir = path.join(args.preprocessed_root, samplename)
    os.makedirs(os.path.dirname(fulldir), exist_ok=True)
    if not os.path.exists(os.path.dirname(fulldir)):
        os.makedirs(os.path.dirname(fulldir))

    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]

    bbxs = list()
    for fb in batches:
        preds = fa.get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            if f is None:
                htmp = int((height - 96) / 2)
                wtmp = int((width - 96) / 2)
                x1, y1, x2, y2 = wtmp, htmp, wtmp + 96, htmp + 96
            # htmp = int((height - 200)/2)
            # wtmp = int((width - 150)/2)
            # x1, y1, x2, y2 = wtmp, htmp, wtmp+150, htmp+200
            else:
                x1, y1, x2, y2 = f
            bbxs.append([x1, y1, x2, y2])
    bbxs = np.array(bbxs)
    np.save(fulldir + '.npy', bbxs)


def main(args, fa):
    print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

    with open(args.filelist) as f:
        lines = f.readlines()

    filelist = [line.strip().split()[0] for line in lines]

    nlength = math.ceil(len(filelist) / args.nshard)
    start_id, end_id = nlength * args.rank, nlength * (args.rank + 1)
    filelist = filelist[start_id: end_id]
    print('process {}-{}'.format(start_id, end_id))

    for vfile in tqdm(filelist):
        process_video_file(vfile, args, fa)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
    parser.add_argument('--filelist', help="Path of a file list containing all samples' name", required=True, type=str)
    parser.add_argument("--video_root", help="Root folder of video", required=True, type=str)
    parser.add_argument('--bbx_root', help="Root folder of bounding boxes of faces", required=True, type=str)
    parser.add_argument("--rank", help="the rank of the current thread in the preprocessing ", default=1, type=str)
    parser.add_argument("--nshard", help="How many threads are used in the preprocessing ", default=1, type=str)

    args = parser.parse_args()

    if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
        raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
    							before running this script!')

    args.rank -= 1
    fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(args.rank))

    main(args, fa)

