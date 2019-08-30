import mxnet as mx


class MxAlexNet(object):
    @staticmethod
    def build(classes):
        data = mx.sym.Variable('data')

        # First block: CONV => RELU => POOL
        convolution_1_1 = mx.sym.Convolution(data=data, kernel=(11, 11), stride=(4, 4), num_filter=96)
        activation_1_1 = mx.sym.LeakyReLU(data=convolution_1_1, act_type='elu')
        batch_norm_1_1 = mx.sym.BatchNorm(data=activation_1_1)
        max_pooling_1 = mx.sym.Pooling(data=batch_norm_1_1, pool_type='max', kernel=(3, 3), stride=(2, 2))
        dropout_1 = mx.sym.Dropout(data=max_pooling_1, p=0.25)

        # Second block: CONV => RELU => POOL
        convolution_2_1 = mx.sym.Convolution(data=dropout_1, kernel=(5, 5), pad=(2, 2), num_filter=256)
        activation_2_1 = mx.sym.LeakyReLU(data=convolution_2_1, act_type='elu')
        batch_norm_2_1 = mx.sym.BatchNorm(data=activation_2_1)
        max_pooling_2 = mx.sym.Pooling(data=batch_norm_2_1, pool_type='max', kernel=(3, 3), stride=(2, 2))
        dropout_2 = mx.sym.Dropout(data=max_pooling_2, p=0.25)

        # Third block: (CONV => RELU) * 3 => POOL
        convolution_3_1 = mx.sym.Convolution(data=dropout_2, kernel=(3, 3), pad=(1, 1), num_filter=384)
        activation_3_1 = mx.sym.LeakyReLU(data=convolution_3_1, act_type='elu')
        batch_norm_3_1 = mx.sym.BatchNorm(data=activation_3_1)
        convolution_3_2 = mx.sym.Convolution(data=batch_norm_3_1, kernel=(3, 3), pad=(1, 1), num_filter=384)
        activation_3_2 = mx.sym.LeakyReLU(data=convolution_3_2, act_type='elu')
        batch_norm_3_2 = mx.sym.BatchNorm(data=activation_3_2)
        convolution_3_3 = mx.sym.Convolution(data=batch_norm_3_2, kernel=(3, 3), pad=(1, 1), num_filter=256)
        activation_3_3 = mx.sym.LeakyReLU(data=convolution_3_3, act_type='elu')
        batch_norm_3_3 = mx.sym.BatchNorm(data=activation_3_3)
        max_pooling_3 = mx.sym.Pooling(data=batch_norm_3_3, pool_type='max', kernel=(3, 3), stride=(2, 2))
        dropout_3 = mx.sym.Dropout(data=max_pooling_3, p=0.25)

        # Fourth block: FC => RELU
        flatten = mx.sym.Flatten(data=dropout_3)
        fully_connected_1 = mx.sym.FullyConnected(data=flatten, num_hidden=4096)
        activation_4_1 = mx.sym.LeakyReLU(data=fully_connected_1, act_type='elu')
        batch_norm_4_1 = mx.sym.BatchNorm(data=activation_4_1)
        dropout_4 = mx.sym.Dropout(data=batch_norm_4_1, p=0.5)

        # Fifth block: FC => RELU
        fully_connected_2 = mx.sym.FullyConnected(data=dropout_4, num_hidden=4096)
        activation_5_1 = mx.sym.LeakyReLU(data=fully_connected_2, act_type='elu')
        batch_norm_5_1 = mx.sym.BatchNorm(data=activation_5_1)
        dropout_5 = mx.sym.Dropout(data=batch_norm_5_1, p=0.5)

        # Softmax classifier
        fully_connected_3 = mx.sym.FullyConnected(data=dropout_5, num_hidden=classes)
        model = mx.sym.SoftmaxOutput(data=fully_connected_3, name='softmax')

        return model
