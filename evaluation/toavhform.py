import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_root", help="Root folder of video", required=True, type=str)
    parser.add_argument('--audio_root', help="Root folder of audio", required=True, type=str)

    args = parser.parse_args()

    with open('../datalist/test_partial.tsv') as f:
        lines = f.readlines()

    with open('../datalist/test.tsv', 'w') as f:
        f.write('/\n')
        for line in lines[1:]:
            linel = line.split('\t')
            out = '{}\t{}/{}.mp4\t{}/{}.wav\t{}\t{}'.format(linel[0], args.video_root, linel[0], args.audio_root, linel[0], linel[1], linel[2])
            f.write(out)

