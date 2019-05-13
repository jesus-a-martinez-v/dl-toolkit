from keras.utils import plot_model

from nn.conv.lenet import LeNet

model = LeNet.build(width=28, height=28, depth=1, classes=10)
plot_model(model, to_file='lenet.png', show_shapes=True)
