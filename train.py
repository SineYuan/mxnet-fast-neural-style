import os
import logging

logging.basicConfig(level=logging.DEBUG)
from argparse import ArgumentParser

import mxnet as mx
import numpy as np

from descriptor import get_content_excutor, get_loss_excutor, get_style_excutor, get_tv_grad_executor
from generator import get_module
from utils import exists, preprocess_img, get_img, img_generator

CONTENT_WEIGHT = 150
STYLE_WEIGHT = 1.2
TV_WEIGHT = 1e-2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
CHECKPOINT_DIR = 'checkpoint'
CHECKPOINT_ITERATIONS = 50
VGG_PATH = "data/vgg/vgg19.params"
BATCH_SIZE = 5
GPU = 0


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--style-image', type=str,
                        dest='style_image', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', required=True)

    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in (default %(default)s)',
                        metavar='CHECKPOINT_DIR', default=CHECKPOINT_DIR)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs (default %(default)s)',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size (default %(default)s)',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency (default %(default)s)',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    parser.add_argument('--gpu', type=int, default=GPU,
                        help='which gpu card to use, -1 means using cpu (default %(default)s)')

    return parser


def check_opts(opts):
    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    exists(opts.style_image, "style image path not found!")
    exists(opts.train_path, "train path not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight > 0
    assert opts.learning_rate >= 0


def save_model(mod, style_path, checkpoint_dir, label=None):
    style_file = os.path.basename(os.path.normpath(style_path))
    model_save_path = os.path.join(checkpoint_dir, style_file.split('.')[0])
    if label is not None:
        model_save_path += '-' + str(label)
    print("Save model to", model_save_path)
    mod.save_params(model_save_path + '.params')
    mod.symbol.save(model_save_path + '.json')


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    check_opts(args)

    # init
    ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
    ctx = mx.cpu()
    vgg_params = mx.nd.load(args.vgg_path)

    # init style
    print('load style image', args.style_image)
    style_np = preprocess_img(get_img(args.style_image))
    style_np = np.expand_dims(style_np, 0)
    dshape = style_np.shape

    style_exec = get_style_excutor(vgg_params, dshape, ctx)
    style_exec.data[:] = mx.nd.array(style_np)
    style_exec.executor.forward()
    style_array = [mx.nd.repeat(arr.copyto(ctx), axis=0, repeats=args.batch_size) for arr in
                   style_exec.outputs]
    del style_exec
    #
    TRAIN_SHAPE = (256, 256)
    dshape = (args.batch_size, 3, *TRAIN_SHAPE)

    # content
    content_exec = get_content_excutor(vgg_params, dshape, ctx)

    # loss
    loss_exec, gscale = get_loss_excutor(vgg_params, dshape, ctx)
    for i in range(len(style_array)):
        loss_exec.arg_dict["target_gram_%d" % i][:] = style_array[i]

    grad_array = []
    for i in range(len(style_array)):
        grad_array.append(mx.nd.ones((1,), ctx) * (float(args.style_weight) / gscale[i]))
    grad_array.append(mx.nd.ones((1,), ctx) * (float(args.content_weight)))

    print([g.asnumpy() for g in grad_array])

    # generator
    gen = get_module(dshape, ctx)
    gen.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': args.learning_rate,
            'wd': 1e-4,
        })

    file_list = os.listdir(args.train_path)

    for e in range(args.epochs):
        img_iter = img_generator(args.train_path, args.batch_size, TRAIN_SHAPE)
        for i, batch in enumerate(img_iter):
            print('Epoch: %d Batch: %d' % (e, i))

            data = mx.nd.array(batch)
            # get content
            content_exec.data[:] = data
            content_exec.executor.forward()
            content_array = content_exec.executor.outputs[0].copyto(ctx)
            loss_exec.arg_dict['target_content'][:] = content_array

            # gen forward
            gen.forward(mx.io.DataBatch([data], [0]), is_train=True)

            loss_exec.data[:] = gen.get_outputs()[0]
            loss_exec.executor.forward(is_train=True)
            loss_exec.executor.backward(grad_array)

            grad = loss_exec.data_grad

            if args.tv_weight > 0:
                tv_grad_executor = get_tv_grad_executor(gen.get_outputs()[0], ctx, args.tv_weight)
                tv_grad_executor.forward()
                grad += tv_grad_executor.outputs[0].copyto(ctx)

            gen.backward([grad])
            gen.update()
            print('update')
            if (i + 1) % args.checkpoint_iterations == 0:
                # save_model(gen, args.style_image, args.checkpoint_dir, label='e%d-i%d' % (e, i))
                save_model(gen, args.style_image, args.checkpoint_dir, label='checkpoint')

        save_model(gen, args.style_image, args.checkpoint_dir, label='checkpoint')

    print("Train Done!")
    save_model(gen, args.style_image, args.checkpoint_dir)
