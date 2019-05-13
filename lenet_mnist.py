import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from nn.conv.lenet import LeNet

print('[INFO] Accessing MNIST...')
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    train_data = train_data.reshape((train_data.shape[0], 1, 28, 28))
    test_data = test_data.reshape((test_data.shape[0], 1, 28, 28))
else:
    train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
    test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

label_binarizer = LabelBinarizer()
train_labels = label_binarizer.fit_transform(train_labels)
test_labels = label_binarizer.fit_transform(test_labels)

print('[INFO] Compiling model...')
optimizer = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print('[INFO] Training network...')
H = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), batch_size=128, epochs=20)

print('[INFO] Evaluating network...')
predictions = model.predict(test_data, batch_size=128)
target_names = [str(x) for x in label_binarizer.classes_]
print(classification_report(test_labels.argmax(axis=1), predictions.argmax(axis=1), target_names=target_names))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 20), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 20), H.history['val_loss'], label='vañ_loss')
plt.plot(np.arange(0, 20), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, 20), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
