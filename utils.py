import os
import logging
import random

import numpy as np
from skimage import io, transform

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)


def exists(p, msg):
    assert os.path.exists(p), msg


def postprocess_img(im):
    #     im = im[0]
    im[0, :] += 123.68
    im[1, :] += 116.779
    im[2, :] += 103.939
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 0, 1)
    im = np.clip(im, 0, 255)
    return im.astype(np.uint8)


def crop_img(im, size):
    if im.shape[0] * size[1] > im.shape[1] * size[0]:
        c = (im.shape[0] - 1. * im.shape[1] / size[1] * size[0]) / 2
        c = int(c)
        im = im[c:-(1 + c), :, :]
    else:
        c = (im.shape[1] - 1. * im.shape[0] / size[0] * size[1]) / 2
        c = int(c)
        im = im[:, c:-(1 + c), :]
    im = transform.resize(im, size)
    im *= 255
    return im


def preprocess_img(im):
    im = im.astype(np.float32)
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 1, 2)
    im[0, :] -= 123.68
    im[1, :] -= 116.779
    im[2, :] -= 103.939
    return im


def get_img(img_path, size=None):
    im = io.imread(img_path)
    if size is not None:
        im = crop_img(im, size)
    return im.astype(np.float64)


def save_output(gen, dest):
    out = gen.get_outputs()[0]
    io.imsave(dest, postprocess_img(out.asnumpy()[0]))


def img_generator(train_path, batch_size, size=None):
    file_list = os.listdir(train_path)
    random.shuffle(file_list)

    batch = []
    for idx in range(len(file_list)):
        if batch_size > len(file_list) - idx:
            break

        img_path = os.path.join(train_path, file_list[idx])
        print("load image", img_path)

        try:
            im = preprocess_img(get_img(img_path, size=size))
        except Exception as e:
            print(e)
            continue
        batch.append(im)
        if len(batch) == batch_size:
            yield np.array(batch)
            # yield batch
            batch = []
