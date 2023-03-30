import os
import math
import argparse
import subprocess
from tqdm import tqdm


def main(args):

    with open(args.filelist) as f:
        lines = f.readlines()

    nlength = math.ceil(len(lines) / args.nshard)
    start_id, end_id = nlength * args.rank, nlength * (args.rank + 1)
    print('process {}-{}'.format(start_id, end_id))

    for line in tqdm(lines[start_id: end_id]):
        path = line.strip()

        video_path = '{}/{}.mp4'.format(args.video_root, path)
        wav_path = '{}/{}.wav'.format(args.save_root, path)

        if not os.path.exists(os.path.dirname(wav_path)):
            os.makedirs(os.path.dirname(wav_path))
        cmd = "ffmpeg -i " + video_path + " -f wav -vn -y " + wav_path + ' -loglevel quiet'

        subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filelist', help="Path of a file list containing all samples' name", required=True, type=str)
    parser.add_argument("--video_root", help="Root folder of video", required=True, type=str)
    parser.add_argument('--audio_root', help="Root folder of saving audio", required=True, type=str)
    parser.add_argument("--rank", help="the rank of the current thread in the preprocessing ", default=1, type=str)
    parser.add_argument("--nshard", help="How many threads are used in the preprocessing ", default=1, type=str)

    args = parser.parse_args()

    args.rank -= 1

    main(args)







