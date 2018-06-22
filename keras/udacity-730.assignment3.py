from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from six.moves import cPickle as pickle

batch_size = 128
num_classes = 10
epochs = 1 #12
img_rows, img_cols = 28, 28
input_shape = img_rows * img_cols


## Load and prepare data ##################################

# Load nonMNIST data
pickle_file = '../udacity-730/data/notMNIST.python2.pickle'
with open(pickle_file, 'rb') as f:
  datasets = pickle.load(f)
x_train = datasets['train_dataset']
y_train = datasets['train_labels']
x_valid = datasets['valid_dataset']
y_valid = datasets['valid_labels']
x_test = datasets['test_dataset']
y_test = datasets['test_labels']
del datasets  # hint to help gc free up memory

# Reshape features to each be 1-D array of length 28*28 = 784
x_train = x_train.reshape(x_train.shape[0], input_shape)
x_valid = x_valid.reshape(x_valid.shape[0], input_shape)
x_test = x_test.reshape(x_test.shape[0], input_shape)
x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print('x_valid shape:', x_valid.shape)
print('x_test shape:', x_test.shape)

# Convert class vectors to binary class matrices  (one hot encoding)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_valid shape:', y_valid.shape)
print('y_test shape:', y_test.shape)

###########################################################



## Build model ############################################

model = Sequential()
model.add(Dense(1024, activation='relu', input_dim = input_shape))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu', input_dim = input_shape))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

###########################################################



## Train model ############################################

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

###########################################################
