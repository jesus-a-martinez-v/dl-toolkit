import argparse
import os

import numpy as np
from imutils import paths
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from datasets.simpledatasetloader import SimpleDatasetLoader
from nn.conv.fcheadnet import FCHeadNet
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-d', '--dataset', required=True, help='Path to input dataset.')
argument_parser.add_argument('-m', '--model', required=True, help='Path to output model.')
arguments = vars(argument_parser.parse_args())

augmenter = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                               zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

print('[INFO] Loading images...')
image_paths = list(paths.list_images(arguments['dataset']))
class_names = [pt.split(os.path.sep)[-2] for pt in image_paths]
class_names = [str(x) for x in np.unique(class_names)]

aspect_aware_preprocessor = AspectAwarePreprocessor(224, 224)
image_to_array_preprocessor = ImageToArrayPreprocessor()

simple_data_loader = SimpleDatasetLoader(preprocessors=[aspect_aware_preprocessor, image_to_array_preprocessor])
data, labels = simple_data_loader.load(image_paths, verbose=500)
data = data.astype('float') / 255.0

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
head_model = FCHeadNet.build(base_model, len(class_names), 256)

model = Model(inputs=base_model.input, outputs=head_model)

for layer in base_model.layers:
    layer.trainable = False

print('[INFO] Compiling model...')
optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print('[INFO] Training head...')
model.fit_generator(augmenter.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=25,
                    steps_per_epoch=len(X_train) // 32, verbose=1)

print('[INFO] Evaluating after initialization...')
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=class_names))

for layer in base_model.layers[15:]:
    layer.trainable = True

print('[INFO] Re-compiling model...')
optimizer = SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print('[INFO] Fine-tuning model...')
model.fit_generator(augmenter.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=100,
                    steps_per_epoch=len(X_train) // 32, verbose=1)

print('[INFO] Evaluating after fine-tuning...')
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=class_names))

print('[INFO] Serializing model...')
model.save(arguments['model'])
