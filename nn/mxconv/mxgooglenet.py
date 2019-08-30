import mxnet as mx


class MxGoogLeNet(object):
    @staticmethod
    def conv_module(data, K, k_X, k_Y, pad=(0, 0), stride=(1, 1)):
        # Defines the CONV => BN => RELU pattern
        conv = mx.sym.Convolution(data=data, kernel=(k_X, k_Y), num_filter=K, pad=pad, stride=stride)
        bn = mx.sym.BatchNorm(data=conv)
        act = mx.sym.Activation(data=bn, act_type='relu')

        return act

    @staticmethod
    def inception_module(data, num_1x1, num_3x3_reduce, num_3x3, num_5x5_reduce, num_5x5, num_1x1_proj):
        # The first branch of the Inception module consists of 1x1 convolutions
        conv_1x1 = MxGoogLeNet.conv_module(data, num_1x1, 1, 1)

        # The second branch of the Inception module is a set of 1x1 convolutions followed by 3x3 convolutions
        conv_r3x3 = MxGoogLeNet.conv_module(data, num_3x3_reduce, 1, 1)
        conv_3x3 = MxGoogLeNet.conv_module(conv_r3x3, num_3x3, 3, 3, pad=(1, 1))

        # The third branch of the Inception module is a set of 1x1 convolutions followed by 5x5 convolutions.
        conv_r5x5 = MxGoogLeNet.conv_module(data, num_5x5_reduce, 1, 1)
        conv_5x5 = MxGoogLeNet.conv_module(conv_r5x5, num_5x5, 5, 5, pad=(2, 2))

        # The final branch of the Inception module is the POOL + projection layer set.
        pool = mx.sym.Pooling(data=data, pool_type='max', pad=(1, 1), kernel=(3, 3), stride=(1, 1))
        conv_proj = MxGoogLeNet.conv_module(pool, num_1x1_proj, 1, 1)

        # Concatenate the filters across the channel dimension
        concat = mx.sym.Concat(*[conv_1x1, conv_3x3, conv_5x5, conv_proj])

        return concat

    @staticmethod
    def build(classes):
        data = mx.sym.Variable('data')

        # First block: CONV => POOL => CONV => CONV => POOL
        conv1_1 = MxGoogLeNet.conv_module(data, 64, 7, 7, pad=(3, 3), stride=(2, 2))
        pool1 = mx.sym.Pooling(data=conv1_1, pool_type='max', pad=(1, 1), kernel=(3, 3), stride=(2, 2))
        conv1_2 = MxGoogLeNet.conv_module(pool1, 64, 1, 1)
        conv1_3 = MxGoogLeNet.conv_module(conv1_2, 192, 3, 3, pad=(1, 1))
        pool2 = mx.sym.Pooling(data=conv1_3, pool_type='max', pad=(1, 1), kernel=(3, 3), stride=(2, 2))

        # Third block: (INCEP * 2) => POOL
        in_3a = MxGoogLeNet.inception_module(pool2, 64, 96, 128, 16, 32, 32)
        in_3b = MxGoogLeNet.inception_module(in_3a, 128, 128, 192, 32, 96, 64)
        pool3 = mx.sym.Pooling(data=in_3b, pool_type='max', pad=(1, 1), kernel=(3, 3), stride=(2, 2))

        # Fourth block: (INCEP * 5) => POOL
        in_4a = MxGoogLeNet.inception_module(pool3, 192, 96, 208, 16, 48, 64)
        in_4b = MxGoogLeNet.inception_module(in_4a, 160, 112, 224, 24, 64, 64)
        in_4c = MxGoogLeNet.inception_module(in_4b, 128, 128, 256, 24, 64, 64)
        in_4d = MxGoogLeNet.inception_module(in_4c, 112, 144, 288, 32, 64, 64)
        in_4e = MxGoogLeNet.inception_module(in_4d, 256, 160, 320, 32, 128, 128)
        pool4 = mx.sym.Pooling(data=in_4e, pool_type='max', pad=(1, 1), kernel=(3, 3), stride=(2, 2))

        # Fifth block: (INCEP * 2) => POOL => DROPOUT
        in_5a = MxGoogLeNet.inception_module(pool4, 256, 160, 320, 32, 128, 128)
        in_5b = MxGoogLeNet.inception_module(in_5a, 384, 192, 384, 48, 128, 128)
        pool5 = mx.sym.Pooling(data=in_5b, pool_type='avg', kernel=(7, 7), stride=(1, 1))
        do = mx.sym.Dropout(data=pool5, p=0.4)

        # Softmax classifier
        flatten = mx.sym.Flatten(data=do)
        fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=classes)
        model = mx.sym.SoftmaxOutput(data=fc1, name='softmax')

        return model
