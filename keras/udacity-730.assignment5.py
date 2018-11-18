%matplotlib inline
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras import backend as K
from keras.callbacks import TensorBoard

vocabulary_size = 50000



# Extract the first file enclosed in a zip file as a list of words
def read_data():
  filename = '../udacity-730/data/text8.zip'
  with zipfile.ZipFile(filename) as f:
    words = tf.compat.as_str(f.read(f.namelist()[0])).split()
  print('Data size %d' % len(words))
  return words


# Build the dictionary and replace rare words with UNK token.
def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary



## Main program ##################################

if 'words' and 'dictionary' not in vars():
  words = read_data()

if 'dictionary' not in vars():
  data, count, dictionary, reverse_dictionary = build_dataset(words)
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10])
  del words  # Hint to reduce memory.



