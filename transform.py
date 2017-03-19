import os
from argparse import ArgumentParser

import mxnet as mx
import numpy as np

from generator import get_module
from utils import exists, get_img, save_output, preprocess_img

GPU = 0


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint',
                        help='checkpoint params file which generated in training',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--in-path', type=str,
                        dest='in_path', help='dir or file to transform',
                        metavar='IN_PATH', required=True)

    parser.add_argument('--out-path', type=str, dest='out_path',
                        help='destination dir of transformed file or files',
                        metavar='OUT_PATH',
                        required=True)

    parser.add_argument('--resize', type=int, nargs=2, dest='resize',
                        help='resize the input image files, usage: --resize=300 400',
                        )

    parser.add_argument('--gpu', type=int, default=GPU,
                        help='which gpu card to use, -1 means using cpu (default %(default)s)')

    return parser


def check_opts(opts):
    exists(opts.checkpoint, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')


def transform_img(img_path, out_dir, img_size):
    img_np = preprocess_img(get_img(img_path, size=img_size))
    img_np = np.expand_dims(img_np, 0)

    # generator
    gen = get_module(img_np.shape, ctx)
    gen.load_params(args.checkpoint)

    data = mx.nd.array(img_np)
    gen.forward(mx.io.DataBatch([data], [0]), is_train=False)

    save_file = os.path.basename(os.path.normpath(img_path))
    save_output(gen, os.path.join(out_dir, save_file))


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    check_opts(args)

    ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()

    if not os.path.isdir(args.in_path):
        transform_img(args.in_path, args.out_path, args.resize)
    else:
        file_list = os.listdir(args.in_path)
        for i in range(len(file_list)):
            transform_img(os.path.join(args.in_path, file_list[i]), args.out_path, args.resize)
