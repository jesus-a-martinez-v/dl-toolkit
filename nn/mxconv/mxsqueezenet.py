import mxnet as mx


class MxSqueezeNet(object):
    @staticmethod
    def squeeze(input, num_filter):
        # The first part of the FIRE module consists of a number of 1x1 filter squeezes
        # on the input data followed by an activation.
        squeeze_1x1 = mx.sym.Convolution(data=input, kernel=(1, 1), stride=(1, 1), num_filter=num_filter)
        act_1x1 = mx.sym.LeakyReLU(data=squeeze_1x1, act_type='elu')

        return act_1x1

    @staticmethod
    def fire(input, num_squeeze_filter, num_expand_filter):
        squeeze_1x1 = MxSqueezeNet.squeeze(input, num_squeeze_filter)
        expand_1x1 = mx.sym.Convolution(data=squeeze_1x1, kernel=(1, 1), stride=(1, 1), num_filter=num_expand_filter)
        relu_expand_1x1 = mx.sym.LeakyReLU(data=expand_1x1, act_type='elu')

        expand_3x3 = mx.sym.Convolution(data=squeeze_1x1, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                        num_filter=num_expand_filter)
        relu_expand_3x3 = mx.sym.LeakyReLU(data=expand_3x3, act_type='elu')

        # The output of the FIRE module is the concatenation of the activation
        # for the 1x1 and 3x3 expands along the channel dimension.
        output = mx.sym.Concat(relu_expand_1x1, relu_expand_3x3, dim=1)

        return output

    @staticmethod
    def build(classes):
        data = mx.sym.Variable('data')

        # First block: CONV => RELU => POOL
        conv_1 = mx.sym.Convolution(data=data, kernel=(7, 7), stride=(2, 2), num_filter=96)
        relu_1 = mx.sym.LeakyReLU(data=conv_1, act_type='elu')
        pool_1 = mx.sym.Pooling(data=relu_1, kernel=(3, 3), stride=(2, 2), pool_type='max')

        # Second to fourth blocks: (FIRE * 3) => POOL
        fire_2 = MxSqueezeNet.fire(pool_1, num_squeeze_filter=16, num_expand_filter=64)
        fire_3 = MxSqueezeNet.fire(fire_2, num_squeeze_filter=16, num_expand_filter=64)
        fire_4 = MxSqueezeNet.fire(fire_3, num_squeeze_filter=32, num_expand_filter=128)
        pool_4 = mx.sym.Pooling(data=fire_4, kernel=(3, 3), stride=(2, 2), pool_type='max')

        # Fifth to eight blocks: (FIRE * 4) => POOL
        fire_5 = MxSqueezeNet.fire(pool_4, num_squeeze_filter=32, num_expand_filter=128)
        fire_6 = MxSqueezeNet.fire(fire_5, num_squeeze_filter=48, num_expand_filter=192)
        fire_7 = MxSqueezeNet.fire(fire_6, num_squeeze_filter=48, num_expand_filter=192)
        fire_8 = MxSqueezeNet.fire(fire_7, num_squeeze_filter=64, num_expand_filter=256)
        pool_8 = mx.sym.Pooling(data=fire_8, kernel=(3, 3), stride=(2, 2), pool_type='max')

        # Nineth and tenth blocks:
        fire_9 = MxSqueezeNet.fire(pool_8, num_squeeze_filter=64, num_expand_filter=256)
        do_9 = mx.sym.Dropout(data=fire_9, p=0.5)
        conv_10 = mx.sym.Convolution(data=do_9, num_filter=classes, kernel=(1, 1), stride=(1, 1))
        relu_10 = mx.sym.LeakyReLU(data=conv_10, act_type='elu')
        pool_10 = mx.sym.Pooling(data=relu_10, kernel=(13, 13), pool_type='avg')

        flatten = mx.sym.Flatten(data=pool_10)
        model = mx.sym.SoftmaxOutput(data=flatten, name='softmax')

        return model
