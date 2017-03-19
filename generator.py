import mxnet as mx


def Conv(data, num_filter, kernel=(5, 5), pad=(2, 2), stride=(2, 2)):
    sym = mx.sym.Convolution(data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=False)
    sym = mx.sym.BatchNorm(sym, fix_gamma=False)
    sym = mx.sym.LeakyReLU(sym, act_type="leaky")
    return sym


def Deconv(data, num_filter, im_hw, kernel=(7, 7), pad=(2, 2), stride=(2, 2), crop=True, out=False):
    sym = mx.sym.Deconvolution(data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True)
    if crop:
        sym = mx.sym.Crop(sym, offset=(1, 1), h_w=im_hw, num_args=1)
    sym = mx.sym.BatchNorm(sym, fix_gamma=False)
    if out == False:
        sym = mx.sym.LeakyReLU(sym, act_type="leaky")
    else:
        sym = mx.sym.Activation(sym, act_type="tanh")
    return sym


def get_module(dshape, ctx, is_train=True):
    sym = generator_symbol()
    mod = mx.mod.Module(symbol=sym,
                        data_names=("data",),
                        label_names=None,
                        context=ctx)
    if is_train:
        mod.bind(data_shapes=[("data", dshape)], for_training=True, inputs_need_grad=True)
    else:
        mod.bind(data_shapes=[("data", dshape)], for_training=False, inputs_need_grad=False)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    return mod


def block(data, num_filter, name):
    data2 = conv(data, num_filter, 1, name=name)
    data2 = mx.sym.Convolution(data=data2, num_filter=num_filter, kernel=(3, 3), pad=(1, 1), name='%s_conv1' % name)
    data2 = mx.sym.InstanceNorm(data=data2, name='%s_in1' % name)
    return mx.sym.Activation(data=data + data2, act_type='relu')


def conv(data, num_filter, stride, name):
    data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3, 3), pad=(1, 1), stride=(stride, stride),
                              name='%s_conv' % name)
    data = mx.sym.InstanceNorm(data=data, name='%s_conv' % name)
    data = mx.sym.Activation(data=data, act_type='relu')
    return data


def generator_symbol():
    data = mx.sym.Variable('data')
    data = mx.sym.Convolution(data=data, num_filter=32, kernel=(9, 9), pad=(4, 4), name='conv0')
    data = mx.sym.InstanceNorm(data=data, name='in0')
    data = mx.sym.Activation(data=data, act_type='relu')
    data = conv(data, 64, 2, name='downsample0')
    data = conv(data, 128, 2, name='downsample1')
    data = block(data, 128, name='block0')
    data = block(data, 128, name='block1')
    data = block(data, 128, name='block2')
    data = block(data, 128, name='block3')
    data = block(data, 128, name='block4')
    data = mx.sym.Deconvolution(data=data, kernel=(4, 4), pad=(0, 0), stride=(2, 2), num_filter=64, name='deconv0')
    data = mx.sym.InstanceNorm(data=data, name='dcin0')
    data = mx.sym.Activation(data=data, act_type='relu')
    data = mx.sym.Deconvolution(data=data, kernel=(4, 4), pad=(0, 0), stride=(2, 2), num_filter=32, name='deconv1')
    data = mx.sym.InstanceNorm(data=data, name='dcin1')
    data = mx.sym.Activation(data=data, act_type='relu')
    data = mx.sym.Convolution(data=data, num_filter=3, kernel=(9, 9), pad=(1, 1), name='lastconv')
    return data
