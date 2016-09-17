# From Deep Learning course https://classroom.udacity.com/courses/ud730

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import random
import urllib.request


################################################################################
# Load the datasets from pickle file
################################################################################

pickle_file = 'notMNIST.pickle'

# with open(pickle_file, 'rb') as f:
#   datasets = pickle.load(f)
#   train_dataset = datasets['train_dataset']
#   train_labels = datasets['train_labels']
#   valid_dataset = datasets['valid_dataset']
#   valid_labels = datasets['valid_labels']
#   test_dataset = datasets['test_dataset']
#   test_labels = datasets['test_labels']
#   del datasets  # hint to help gc free up memory
#   print('Training set', train_dataset.shape, train_labels.shape)
#   print('Validation set', valid_dataset.shape, valid_labels.shape)
#   print('Test set', test_dataset.shape, test_labels.shape)

s3_pickle_url = "https://s3.amazonaws.com/ml-playground/" + pickle_file
print(s3_pickle_url)
datasets = pickle.load(urllib.request.urlopen(s3_pickle_url))
train_dataset = datasets['train_dataset']
train_labels = datasets['train_labels']
valid_dataset = datasets['valid_dataset']
valid_labels = datasets['valid_labels']
test_dataset = datasets['test_dataset']
test_labels = datasets['test_labels']
del datasets  # hint to help gc free up memory
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)



################################################################################
# Reformat the data:  flatten and 1-hot encodings
################################################################################

IMAGE_SIZE = 28
NUM_LABELS = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)




################################################################################
# Measure the accuracy of predictions
################################################################################

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])





################################################################################
# Define network
################################################################################

BATCH_SIZE = 128
SAMPLE_SIZE = 10

def forward_prop(dataset, train=False):
  layer1 = tf.matmul(dataset, weights1) + biases1
  keep_prob = 0.5 if train else 1.0
  relu = tf.nn.dropout(tf.nn.relu(layer1), keep_prob)
  return tf.matmul(relu, weights2) + biases2

def loss():
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  regularizers = (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(biases1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(biases2))
  loss += 5e-4 * regularizers
  return loss


graph5 = tf.Graph()
with graph5.as_default():

  # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.as_dtype(train_dataset.dtype), shape = (BATCH_SIZE, train_dataset.shape[1]))
  tf_train_labels = tf.placeholder(tf.as_dtype(train_labels.dtype), shape = (BATCH_SIZE, train_labels.shape[1]))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Weights & Biases
  input_size = train_dataset.shape[1]
  hidden_size = 1028
  output_size = train_labels.shape[1]
  weights1 = tf.Variable(tf.truncated_normal([input_size, hidden_size]))
  biases1 = tf.Variable(tf.truncated_normal([hidden_size], stddev=1.0))
  weights2 = tf.Variable(tf.truncated_normal([hidden_size, output_size]))
  biases2 = tf.Variable(tf.truncated_normal([output_size], stddev=1.0))

  # Training computation.
  logits = forward_prop(tf_train_dataset, train=True)
  loss = loss()

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(forward_prop(tf_valid_dataset))
  test_prediction = tf.nn.softmax(forward_prop(tf_test_dataset))

print('network defined')



################################################################################
# Train and predict
################################################################################

num_training_examples = train_dataset.shape[0]

with tf.Session(graph=graph5) as session:
  tf.initialize_all_variables().run()

  for step in range(3001):
    offset = (step % SAMPLE_SIZE * BATCH_SIZE) % (num_training_examples - BATCH_SIZE)

    batch_dataset = train_dataset[offset:(offset+BATCH_SIZE), :]
    batch_labels = train_labels[offset:(offset+BATCH_SIZE), :]

    feed_dict = {tf_train_dataset: batch_dataset, tf_train_labels: batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

    if step % 500 == 0:
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels), "\n")

  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
