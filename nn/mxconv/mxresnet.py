import mxnet as mx


class MxResNet(object):
    @staticmethod
    def residual_module(data, K, stride, reduce=False, batch_norm_epsilon=2e-5, batch_norm_momentum=0.9):
        shortcut = data

        # The first block of the ResNet module are 1x1 CONVs.
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=batch_norm_epsilon, momentum=batch_norm_momentum)
        act1 = mx.sym.Activation(data=bn1, act_type='relu')
        conv1 = mx.sym.Convolution(data=act1, pad=(0, 0), kernel=(1, 1), stride=(1, 1), num_filter=int(K * 0.25),
                                   no_bias=True)

        # The second block of the ResNet module are 3x3 CONVs.
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=batch_norm_epsilon, momentum=batch_norm_momentum)
        act2 = mx.sym.Activation(data=bn2, act_type='relu')
        conv2 = mx.sym.Convolution(data=act2, pad=(1, 1), kernel=(3, 3), stride=stride, num_filter=int(K * 0.25),
                                   no_bias=True)

        # The third block of the ResNet module is another set of 1x1 CONVs
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=batch_norm_epsilon, momentum=batch_norm_momentum)
        act3 = mx.sym.Activation(data=bn3, act_type='relu')
        conv3 = mx.sym.Convolution(data=act3, pad=(0, 0), kernel=(1, 1), stride=(1, 1), num_filter=K, no_bias=True)

        # If we are to reduce the spatial size, apply a CONV layer to the shortcut
        if reduce:
            shortcut = mx.sym.Convolution(data=act1, pad=(0, 0), kernel=(1, 1), stride=stride, num_filter=K,
                                          no_bias=True)

        add = conv3 + shortcut

        return add

    @staticmethod
    def build(classes, stages, filters, batch_norm_epsilon=2e-5, batch_norm_momentum=0.9):
        data = mx.sym.Variable('data')

        # First block: BN => CONV => ACT => POOL, then initialize the body of the network
        bn1_1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=batch_norm_epsilon, momentum=batch_norm_momentum)
        conv1_1 = mx.sym.Convolution(data=bn1_1, pad=(3, 3), kernel=(7, 7), stride=(2, 2), num_filter=filters[0],
                                     no_bias=True)
        bn1_2 = mx.sym.BatchNorm(data=conv1_1, fix_gamma=False, eps=batch_norm_epsilon, momentum=batch_norm_momentum)
        act1_2 = mx.sym.Activation(data=bn1_2, act_type='relu')
        pool1 = mx.sym.Pooling(data=act1_2, pool_type='max', pad=(1, 1), kernel=(3, 3), stride=(2, 2))

        body = pool1

        # Loop over the stages
        for i in range(len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            body = MxResNet.residual_module(body, filters[i + 1], stride, reduce=True,
                                            batch_norm_epsilon=batch_norm_epsilon,
                                            batch_norm_momentum=batch_norm_momentum)

            # Loop over the layers in the current stage
            for j in range(stages[i] - 1):
                body = MxResNet.residual_module(body, filters[i + 1], (1, 1),
                                                batch_norm_epsilon=batch_norm_epsilon,
                                                batch_norm_momentum=batch_norm_momentum)

        bn2_1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=batch_norm_epsilon, momentum=batch_norm_momentum)
        act2_1 = mx.sym.Activation(data=bn2_1, act_type='relu')
        pool2 = mx.sym.Pooling(data=act2_1, pool_type='avg', global_pool=True, kernel=(7, 7))

        flatten = mx.sym.Flatten(data=pool2)
        fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=classes)

        model = mx.sym.SoftmaxOutput(data=fc1, name='softmax')

        return model
