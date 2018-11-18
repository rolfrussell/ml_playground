from __future__ import print_function
from six.moves import cPickle as pickle

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras import backend as K
from keras.callbacks import TensorBoard

batch_size = 128
num_classes = 10
epochs = 3   #12
img_rows, img_cols = 28, 28
num_channels = 1 # grayscale
input_shape = (img_rows, img_cols, num_channels)  # for 2D convolutions


## Load and prepare data ##################################

if 'x_train' not in vars(): \

  # Load nonMNIST data
  pickle_file = '../udacity-730/data/notMNIST.pickle'
  with open(pickle_file, 'rb') as f:
    datasets = pickle.load(f) \

  x_train = datasets['train_dataset']
  y_train = datasets['train_labels']
  x_valid = datasets['valid_dataset']
  y_valid = datasets['valid_labels']
  x_test = datasets['test_dataset']
  y_test = datasets['test_labels']
  del datasets \

  # Optionally flatten features to each be 1-D array of length 28*28 = 784
  x_train = x_train.reshape(x_train.shape[0], *input_shape)
  x_valid = x_valid.reshape(x_valid.shape[0], *input_shape)
  x_test = x_test.reshape(x_test.shape[0], *input_shape)
  print('x_train shape:', x_train.shape)
  print('x_valid shape:', x_valid.shape)
  print('x_test shape:', x_test.shape) \

  # float32
  x_train = x_train.astype('float32')
  x_valid = x_valid.astype('float32')
  x_test = x_test.astype('float32') \

  # Convert class vectors to binary class matrices  (one hot encoding)
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_valid = keras.utils.to_categorical(y_valid, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  print('y_train shape:', y_train.shape)
  print('y_valid shape:', y_valid.shape)
  print('y_test shape:', y_test.shape) \

  # Mini training set
  mini_sample_size = 10*batch_size
  mini_x_train = x_train[:mini_sample_size]
  mini_y_train = y_train[:mini_sample_size]



## Build model ############################################

model = Sequential()
model.add(Conv2D(16, input_shape=input_shape,
                 kernel_size=(5, 5),
                 strides=(1, 1),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16,
                 kernel_size=(5, 5),
                 strides=(1, 1),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))




## Train model ############################################

tensorboard = TensorBoard(log_dir='tensorflow_logs', histogram_freq=0, batch_size=batch_size)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.2, momentum=0.25, decay=0.00, nesterov=False),
              # optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid),
          callbacks=[tensorboard])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

