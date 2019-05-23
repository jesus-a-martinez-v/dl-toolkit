import matplotlib
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from nn.conv.minivggnet import MiniVGGNet

matplotlib.use('Agg')

import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-o', '--output', required=True, help='Path to output directory.')
argument_parser.add_argument('-m', '--models', required=True, help='Path to output models directory.')
argument_parser.add_argument('-n', '--num-models', type=int, default=5, help='Number of models to train.')
arguments = vars(argument_parser.parse_args())

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float') / 255.0
X_test = X_test.astype('float') / 255.0

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

augmenter = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
                               fill_mode='nearest')

for i in range(arguments['num_models']):
    print(f'[INFO] Training model {i + 1}/{arguments["num_models"]}')
    optimizer = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    H = model.fit_generator(augmenter.flow(X_train, y_train, batch_size=64), validation_data=(X_test, y_test),
                            epochs=40, steps_per_epoch=len(X_train) // 64)
    model_path_segments = [arguments['models'], f'model_{i}.model']
    model.save(os.path.sep.join(model_path_segments))

    predictions = model.predict(X_test, batch_size=64)
    report = classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names)

    model_path_segments = [arguments['output'], f'model_{i}.txt']

    with open(os.path.sep.join(model_path_segments), 'w') as f:
        f.write(report)

    model_path_segments = [arguments['output'], f'model_{i}.png']
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, 40), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, 40), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, 40), H.history['acc'], label='train_acc')
    plt.plot(np.arange(0, 40), H.history['val_acc'], label='val_acc')
    plt.title(f'Training Loss and Accuracy for model {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.savefig(os.path.sep.join(model_path_segments))
    plt.close()
