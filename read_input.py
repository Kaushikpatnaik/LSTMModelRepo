'''
Functions and classes for reading data
'''

import numpy as np
import tensorflow as tf
import pandas as pd
import collections
import os
from six.moves.urllib.request import urlretrieve
import zipfile

def load_zip_data(filename):
  '''
  return the zip data as unicode strings
  :param: filename: name of file to open
  :return:
  '''


  f = zipfile.ZipFile(filename, 'r')
  for name in f.namelist():
    return tf.compat.as_str(f.read(name))
  f.close()

def load_csv_file(filename):
  '''
  return data from csv file
  :param: filename: name of file with location
  :return:
  '''

  with open(filename,'r') as f:
    data = f.read()
  f.close()
  return data

def train_test_split(data, ratio =[0.6,0.2,0.2]):
  '''
  Based on ratio, splits the data into train, validation and testing steps
  :param data: data on which the split needs to take place
  :param ratio: a list containing ratio's for train, test and split
  :return: train_data, valid_data and test_data
  '''

  [train_index, valid_index, test_index] = [int(x*len(data)) for x in ratio]
  train_data = data[:train_index]
  valid_data = data[train_index:train_index+valid_index]
  test_data = data[len(data)-valid_index:]

  return (train_data,valid_data,test_data)

class BatchGenerator(object):
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
    self.cursor = 0
    self.segment_len = len(self.data)//self.batch_size
    self.batch = np.zeros((self.batch_size,self.segment_len), dtype=np.int32)
    self.epoch_size = (self.segment_len - 1) // self.batch_len

  def preprocess(self):
    counter = collections.Counter(self.data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    self.chars, _ = zip(*count_pairs)
    self.vocab_size = len(self.chars)
    self.vocab = dict(zip(self.chars, range(self.vocab_size)))
    tensor = np.array(list(map(self.vocab.get, self.data)))
    return tensor

  def create_batches(self):
    # contiguous data should be part of a single batch_len
    # the way to ensure that is to divide the data into segments based on batch_len
    # with segments capturing continuous information
    # and to move along the segments taking consequtive chunks

    tensor = self.preprocess()
    for i in range(self.batch_size):
      self.batch[i] = tensor[self.segment_len*i:self.segment_len*(i+1)]

  def next(self):
    x = self.batch[:,self.cursor*self.batch_len:(self.cursor+1)*self.batch_len]
    y = self.batch[:,self.cursor*self.batch_len+1:(self.cursor+1)*self.batch_len+1]
    self.cursor = (self.cursor + 1)//self.epoch_size
    return (x,y)

if __name__ == "__main__":

  url = 'http://mattmahoney.net/dc/'
  filename = 'text8.zip'

  data = download_data(url,filename)

  print data[:100]

  train, val ,test = train_test_split(data)

  batch_train = BatchGenerator(train,64,20)
  batch_train.create_batches()
  print batch_train.next()



