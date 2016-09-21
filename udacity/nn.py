# From Deep Learning course https://classroom.udacity.com/courses/ud730

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import random
import urllib.request
from sys import platform
import socket
import time


IMAGE_SIZE = 28
NUM_FEATURES = IMAGE_SIZE * IMAGE_SIZE
NUM_LABELS = 10
BATCH_SIZE = 128
START = time.time()


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('s3_data', False, 'If true, loads data from S3.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
flags.DEFINE_integer('max_steps', 3001, 'Number of steps to run trainer.')
flags.DEFINE_integer('epoch_size', 999999999999, 'Size of an epoch, basically how many of the examples to use in training.')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_float('l2_beta', 5e-4, 'L2 regularization beta value.')
flags.DEFINE_float('keep_prob', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', 'tmp/summary_logs', 'Summaries directory')



################################################################################
# Load the datasets from pickle file
# Reformat the data:  flatten and 1-hot encodings
################################################################################
def load_datasets(from_s3 = True):

  def reformat(dataset, labels):
    dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
    return dataset, labels

  pickle_file = 'notMNIST.pickle'
  if FLAGS.s3_data:
    s3_pickle_url = "https://s3.amazonaws.com/ml-playground/" + pickle_file
    datasets = pickle.load(urllib.request.urlopen(s3_pickle_url))
  else:
    with open(pickle_file, 'rb') as f:
      datasets = pickle.load(f)

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
# Build and run model
################################################################################
def train():

  def variable_summaries(var, name):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.scalar_summary('mean/' + name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.scalar_summary('sttdev/' + name, stddev)
      tf.scalar_summary('max/' + name, tf.reduce_max(var))
      tf.scalar_summary('min/' + name, tf.reduce_min(var))
      tf.histogram_summary(name, var)

  def weights_variable(shape):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weights')
    return weights

  def biases_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='biases')


  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
      weights = weights_variable([input_dim, output_dim])
      all_weights.append(weights)
      biases = biases_variable([output_dim])
      pre_activations = tf.matmul(input_tensor, weights) + biases

      if act != None:
        pre_dropouts = act(pre_activations, name='pre_dropouts')
        return tf.nn.dropout(pre_dropouts, keep_prob, name='activations')
      else:
        return pre_activations

  def loss():
    with tf.name_scope('loss'):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')
      loss = tf.reduce_mean(cross_entropy, name='loss')

      l2_regularization = FLAGS.l2_beta * sum(map(tf.nn.l2_loss, all_weights))
      loss += l2_regularization

      tf.scalar_summary('loss', loss)
      return loss

  def accuracy():
    with tf.name_scope('accuracy'):
      predictions = tf.nn.softmax(logits)
      correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
      tf.scalar_summary('accuracy', accuracy)
    return accuracy



  session = tf.Session()

  # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
  with tf.name_scope('input'):
    features = tf.placeholder(tf.float32, shape = (None, NUM_FEATURES), name='features')
    labels = tf.placeholder(tf.float32, shape = (None, NUM_LABELS), name='labels')

  # Define network
  keep_prob = tf.placeholder(tf.float32)
  all_weights = []
  hidden1 = nn_layer(features, input_size, hidden_size, 'layer1')
  logits = nn_layer(hidden1, hidden_size, output_size, 'layer2', act=None)

  # Optimize
  loss = loss()
  optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
  accuracy = accuracy()

  # Merge all the summaries and write them out to file
  merged_summaries = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', session.graph)
  valid_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/valid')
  test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
  tf.initialize_all_variables().run(session=session)


  # Train
  def feed_dict(type, step):
    if type == 'train':
      offset = (step * BATCH_SIZE) % (FLAGS.epoch_size - BATCH_SIZE)
      batch_dataset = train_dataset[offset:(offset+BATCH_SIZE), :]
      batch_labels = train_labels[offset:(offset+BATCH_SIZE), :]
      return {features: batch_dataset, labels: batch_labels, keep_prob: FLAGS.keep_prob}
    elif type == 'valid':
      return {features: valid_dataset, labels: valid_labels, keep_prob: 1.0}
    elif type == 'test':
      return {features: test_dataset, labels: test_labels, keep_prob: 1.0}
    else:
      raise RuntimeError("Don't know data of type ", type, "Was expecting train, valid or test")


  for step in range(FLAGS.max_steps):
    summary, acc1, _ = session.run([merged_summaries, accuracy, optimizer], feed_dict=feed_dict('train', step))
    train_writer.add_summary(summary, step)

    if step % 500 == 0:
      summary, acc = session.run([merged_summaries, accuracy], feed_dict=feed_dict('valid', step))
      valid_writer.add_summary(summary, step)
      print('Step:', step, 'Elapsed seconds:', int(time.time() - START))
      print('Training accuracy:', acc1)
      print('Validation accuracy:', acc, '\n')

  summary, acc = session.run([merged_summaries, accuracy], feed_dict=feed_dict('test', step))
  test_writer.add_summary(summary, step)
  print('Test accuracy:', acc)

  train_writer.close()
  valid_writer.close()
  test_writer.close()
  session.close()



################################################################################
# Execute
################################################################################
print('learning_rate:', FLAGS.learning_rate)
print('l2_beta:', FLAGS.l2_beta)
print('keep_prob:', FLAGS.keep_prob)
print('max_steps', FLAGS.max_steps)
print('epoch_size:', FLAGS.epoch_size)
print('s3_data:', FLAGS.s3_data, '\n')

if 'train_dataset' not in vars():
  train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_datasets()

# use a subset of the training data
train_dataset = train_dataset[:FLAGS.epoch_size]
train_labels = train_labels[:FLAGS.epoch_size]

input_size = train_dataset.shape[1]
hidden_size = 1028
output_size = train_labels.shape[1]

# clear previous training logs
if tf.gfile.Exists(FLAGS.summaries_dir):
  tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
tf.gfile.MakeDirs(FLAGS.summaries_dir)

train()

