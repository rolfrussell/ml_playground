# From Deep Learning course https://classroom.udacity.com/courses/ud730

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import random
import urllib.request
import socket
import time


IMAGE_SIZE = 28
NUM_LABELS = 10
BATCH_SIZE = 128
START = time.time()


################################################################################
# Load & reformat the datasets from pickle file
################################################################################
def load_datasets(from_s3 = True):
  pickle_file = 'notMNIST.pickle'
  if (socket.gethostname() == 'Rolfs-MacBook-Pro.local'):
    with open(pickle_file, 'rb') as f:
      datasets = pickle.load(f)
  else:
    s3_pickle_url = "https://s3.amazonaws.com/ml-playground/" + pickle_file
    datasets = pickle.load(urllib.request.urlopen(s3_pickle_url))

  train_dataset = datasets['train_dataset']
  train_labels = datasets['train_labels']
  valid_dataset = datasets['valid_dataset']
  valid_labels = datasets['valid_labels']
  test_dataset = datasets['test_dataset']
  test_labels = datasets['test_labels']
  del datasets  # hint to help gc free up memory

  train_dataset, train_labels = reformat(train_dataset, train_labels)
  valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
  test_dataset, test_labels = reformat(test_dataset, test_labels)

  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape, '\n')
  return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels



################################################################################
# Reformat the data:  flatten and 1-hot encodings
################################################################################
def reformat(dataset, labels):
  dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
  return dataset, labels



################################################################################
# Measure the accuracy of predictions
################################################################################
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])




################################################################################
# Build and run model
################################################################################
def train():

  def forward_prop(dataset, dropout_keep_prob=1.0):
    layer1 = tf.matmul(dataset, weights1) + biases1
    relu = tf.nn.dropout(tf.nn.relu(layer1), dropout_keep_prob)
    return tf.matmul(relu, weights2) + biases2

  def loss(l2_beta):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    loss += l2_beta * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2))
    return loss


  graph = tf.Graph()
  with graph.as_default():

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
    biases1 = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
    weights2 = tf.Variable(tf.truncated_normal([hidden_size, output_size]))
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_size]))

    # Training computation.
    logits = forward_prop(tf_train_dataset, 0.5)
    loss = loss(5e-4)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(forward_prop(tf_valid_dataset))
    test_prediction = tf.nn.softmax(forward_prop(tf_test_dataset))


  # Train and predict
  num_training_examples = train_dataset.shape[0]

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    for step in range(1001):
      offset = ((step % sample_size) * BATCH_SIZE) % (num_training_examples - BATCH_SIZE)

      batch_dataset = train_dataset[offset:(offset+BATCH_SIZE), :]
      batch_labels = train_labels[offset:(offset+BATCH_SIZE), :]

      feed_dict = {tf_train_dataset: batch_dataset, tf_train_labels: batch_labels}
      _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

      if step % 200 == 0:
        print('Step:', step, 'Elapsed seconds:', int(time.time() - START))
        print("Minibatch loss: %f" % l)
        print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
        print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels), "\n")

    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))



################################################################################
# Execute
################################################################################
if 'train_dataset' not in vars():
  train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_datasets()

sample_size = 100000000

train()
