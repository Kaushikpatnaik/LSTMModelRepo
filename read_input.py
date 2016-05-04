'''
Goal: To re-create the LSTM char to char model from ground up to understand how
LSTMs work in Tensorflow, and move onto better code

Aspiration: To submit a multi-variate time series example on Tensorflow

Step 1: Download the data
Step 2: Convert characters to numbers
Step 3: Reshape data into batches of shape [batch_size, backprop_len]
Step 4: Embeddings, sampling from vector, calculation of log probability
Step 5: LSTM graph
Step 6: Running on training data
Step 7: Test data evaluation

Update 1: Dropout
Update 2: Deep LSTMs
Update 3: Beam Search
Update 4: batch normalization
'''

import numpy as np
import tensorflow as tf
import pandas as pd
import collections
import os
from six.moves.urllib.request import urlretrieve
import zipfile

def download_data(url, filename):
  '''
  Download a zip file if not present, return the data as unicode strings
  :param
  url: url from where to get the data
  filename: name of file to download
  :return:
  '''

  if not os.path.exists(filename):
    filename, _ = urlretrieve(url+filename, filename)

  f = zipfile.ZipFile(filename, 'r')
  for name in f.namelist():
    return tf.compat.as_str(f.read(name))
  f.close()

def char2id(char,vocab):
  '''
  based on vocab, convert characters to numbers
  :param char: character to be converted
  :param vocab: list of characters used as vocabulary
  :return: id for the character
  '''
  if char in vocab:
    return vocab.index(char) + 1
  else:
    return 0

def id2char(id,vocab):
  '''
  based on integer id return the character
  :param id:
  :param vocab:
  :return:
  '''
  if id > 0:
    return vocab[id-1]
  else:
    return ' '

def train_test_split(data, ratio =[0.6,0.2,0.2]):
  '''
  Based on ratio, splits the data into train, validation and testing steps
  :param data: data on which the split needs to take place
  :param ratio: a list containing ratio's for train, test and split
  :return: train_data, valid_data and test_data
  '''
  [train_index, valid_index, test_index] = ratio * len(data)
  train_data = data[:train_index]
  valid_data = data[train_index:train_index+valid_index]
  test_data = data[len(data)-valid_index:]

  return (train_data,valid_data,test_data)

class batch_generator(object):
  '''
  Given a source of data, generate batches with [batch_size, batch_len]
  :param data: data given as a single string of characters. Can be train, test or validation
  :param batch_size: data processed in parallel by the LSTM, stabilizes the variance of SGD
  :param batch_len: the backprop limitation due to vanishing gradients
  :return: a generator/yield function that returns the batches
  '''
  def __init__(self,data,batch_size,batch_len):

    self.data = data
    self.batch_size = batch_size
    self.batch_len = batch_len
    self.segment_len = len(data)//batch_size
    self.vocab_len = 27
    self.cursor = [offset*self.segment_len for offset in range(batch_size)]
    self.last_batch = self._next_batch()

  # contiguous data should be part of a single batch_len
  # the way to ensure that is to divide the data into segments based on batch_len
  # with segments capturing continuous information
  # and to move along the segments taking consequtive chunks

  def _next_batch(self):
    batch = np.zeros((self.batch_size,self.vocab_len))
    for i in range(self.batch_size):
      batch[i,char2id(data[self.cursor[i]])] = 1.0
      # google divides by text_len which is correct ?
      self.cursor[i] = (self.cursor[i]+1) % self.segment_len
    return batch

  def next(self):
    batches = [self.last_batch]
    for step in range(self.batch_len):
      batches.append(self._next_batch())
    self.last_batch = batches[-1]
    return batches

if __name__ == "__main__":

  url = 'http://mattmahoney.net/dc/'
  filename = 'text8.zip'

  data = download_data(url,filename)

  print data[:100]


