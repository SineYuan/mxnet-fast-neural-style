import mxnet as mx
import numpy as np
import model_vgg19 as vgg

from collections import namedtuple

Executor = namedtuple('Executor', ['executor', 'data', 'data_grad', 'arg_dict', 'outputs'])


def style_gram_symbol(input_shape, style):
    _, output_shapes, _ = style.infer_shape(**input_shape)
    gram_list = []
    grad_scale = []
    for i in range(len(style.list_outputs())):
        shape = output_shapes[i]
        x = mx.sym.Reshape(style[i], shape=(int(shape[0]), int(shape[1]), int(np.prod(shape[2:]))))
        gram = mx.sym.batch_dot(x, x, transpose_b=True)
        gram_list.append(gram)
        grad_scale.append(np.prod(shape[1:]) * shape[1])
    return mx.sym.Group(gram_list), grad_scale


def get_loss(gram, content):
    gram_loss = []
    for i in range(len(gram.list_outputs())):
        gvar = mx.sym.Variable("target_gram_%d" % i)
        gram_loss.append(mx.sym.sum(mx.sym.square(gvar - gram[i])))
    cvar = mx.sym.Variable("target_content")
    content_loss = mx.sym.sum(mx.sym.square(cvar - content))
    return mx.sym.Group(gram_loss), content_loss


def get_content_excutor(params, dshape, ctx):
    sytle, content = vgg.get_symbol()
    return vgg.get_executor(content, params, dshape, ctx)


def get_style_excutor(params, dshape, ctx):
    input_shape = {"data": dshape}
    style, content = vgg.get_symbol()
    gram, gscale = style_gram_symbol(input_shape, style)
    return vgg.get_executor(gram, params, dshape, ctx)


def get_loss_excutor(params, dshape, ctx):
    input_shape = {"data": dshape}
    style, content = vgg.get_symbol()
    gram, gscale = style_gram_symbol(input_shape, style)
    style_loss, content_loss = get_loss(gram, content)
    sym = mx.sym.Group([style_loss, content_loss])
    return vgg.get_executor(sym, params, dshape, ctx), gscale


def get_content_loss_excutor(params, dshape, ctx):
    input_shape = {"data": dshape}
    style, content = vgg.get_symbol()
    gram, gscale = style_gram_symbol(input_shape, style)
    style_loss, content_loss = get_loss(gram, content)
    sym = content_loss
    return vgg.get_executor(sym, params, dshape, ctx)


# tv-loss
def get_tv_grad_executor(img, ctx, tv_weight):
    """create TV gradient executor with input binded on img
    """
    if tv_weight <= 0.0:
        return None
    nchannel = img.shape[1]
    simg = mx.sym.Variable("img")
    skernel = mx.sym.Variable("kernel")
    channels = mx.sym.SliceChannel(simg, num_outputs=nchannel)
    out = mx.sym.Concat(*[
        mx.sym.Convolution(data=channels[i], weight=skernel,
                           num_filter=1,
                           kernel=(3, 3), pad=(1, 1),
                           no_bias=True, stride=(1, 1))
        for i in range(nchannel)])
    kernel = mx.nd.array(np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]])
                         .reshape((1, 1, 3, 3)),
                         ctx) / 8.0
    out = out * tv_weight
    return out.bind(ctx, args={"img": img, "kernel": kernel})
