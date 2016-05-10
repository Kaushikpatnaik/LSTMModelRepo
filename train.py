'''
Training file with functions for

1) Taking in the inputs
2) Defining the model
3) Reading the input and generating batches
4) Defining the loss, learning rate and optimization functions
5) Running multiple epochs on training and testing
'''
import argparse
from lstm import *
from read_input import *
from tensorflow.python.ops import math_ops

def sequence_loss_by_example(logits, targets, weights, average_across_time=True, scope=None):
  '''
  A simple version of weighted sequence loss measured in sequence
  :param logits:
  :param targets:
  :param weights:
  :param average_across_time:
  :param softmax_loss_function:
  :param scope:
  :return:
  '''
  if len(logits) != len(targets) or len(weights) != len(logits):
    raise ValueError("Lenghts of logits, weights and target must be same "
                     "%d, %d, %d" %len(logits), len(weights), len(targets))

  with tf.variable_scope(scope or "sequence_loss_by_example"):
    sequence_loss_list = []
    for logit, target, weight in zip(logits, targets, weights):
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logit,target)
      # tensorflow !!!
      sequence_loss_list.append(loss*weight)
    sequence_loss = math_ops.add_n(sequence_loss_list)
    if average_across_time:
      total_weight = math_ops.add_n(weights) + 1e-12
      final_loss = sequence_loss/total_weight
    else:
      final_loss = sequence_loss
    return final_loss

import tensorflow as tf
class Model(object):
  '''Class defining the overall model based on lstm.py'''

  def __init__(self, args):
    self.batch_size = args.batch_size
    self.batch_len = args.batch_len
    self.num_layers = args.num_layers
    self.cell = args.cell
    self.hidden_units = args.hidden_units
    self.data_dim = args.data_dim

    # define placeholder for data layer
    self.input_layer = tf.placeholder(tf.int32, [self.batch_size, self.batch_len])

    # define weights for data layer (vocab_size to self.hidden_units)
    input_weights = tf.get_variable('input_weights', [self.data_dim, self.hidden_units])

    # define input to LSTM cell
    inputs = tf.nn.embedding_lookup(input_weights, self.input_layer)

    # define model based on cell and num_layers
    if self.num_layers ==1:
      lstm_layer = LSTM(self.hidden_units)
    else:
      cells = [LSTM(self.hidden_units)]*self.num_layers
      lstm_layer = DeepLSTM(cells)

    # run the model for multiple time steps
    outputs = []
    initial_state = array_ops.zeros(array_ops.pack([self.batch_size, lstm_layer.state_size]), dtype=tf.float32)
    initial_state.set_shape([None, lstm_layer.state_size])
    state = initial_state
    with tf.variable_scope("RNN"):
      for time in range(len(inputs)):
        if time > 0: tf.get_variable_scope().reuse_variables()
        output, state = lstm_layer(inputs[:,time,:], state)
        outputs.append(output)

    # for each single input collect the hidden units, then reshape with individual time steps * hidden_units
    output = tf.reshape(tf.concat(1,outputs), [-1,self.hidden_units])

    softmax_w = tf.get_variable('softmax_w', [self.hidden_units, self.data_dim])
    softmax_b = tf.get_variable('softmax_b', [self.data_dim])
    # logits is now of shape [self.batch_size * self.batch_len, self.data_dim]
    logits = tf.matmul(output, softmax_w) + softmax_b

    # define placeholder for target layer
    self.targets = tf.placeholder(tf.int32, [self.batch_size, self.batch_len])

    # sequence loss by example
    # to enable comparision by each and every example the row lengths of logits
    # and targets should be same
    loss = sequence_loss_by_example(logits,tf.reshape(self.targets, [-1]),tf.ones([self.batch_size*self.batch_len]))
    self.cost = tf.reduce_sum(loss) / self.batch_size / self.batch_len
    self.final_state = state

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads




















def main():
  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_loc', type=str, default='', help='data location for all data')
  parser.add_argument('--split_ratio', type =list, default=[0.8,0.1,0.1], help='split ratio for train, validation and test')
  parser.add_argument('--batch_size', type=int, default=64, help='batch size for data')
  parser.add_argument('--batch_len', type=int, default=20, help='number of time steps to unroll')
  parser.add_argument('--cell', type=str, default='lstm', help='the cell type to use, currently only LSTM')
  parser.add_argument('--num_layers', type=int, default=1, help='depth of hidden units in the model')
  parser.add_argument('--hidden_units', type=int, default=64, help='number of hidden units in the cell')
  parser.add_argument('--data_dim', type=int, default=27, help='dimension of data, currently default is 27')
  parser.add_argument('--num_epochs', type=int, default = 20, help='max number of epochs to run the training')
  parser.add_argument('--lr_rate', type=float, default=0.001, help='learning rate')
  parser.add_argument('--lr_decay', type=float, default=0.97, help='learning rate decay')
  parser.add_argument('--optim_func', type=str, default='rmsprop', help='optimization function to be used')
  parser.add_argument('--grad_clip', type=float, default=5.0, help='clip gradients at this value')
  parser.add_argument('--save_every', type=int, default=500, help='save at every batches')

  # load data


  # create model
  # TODO Kaushik If Dropout is included, we need to create separate model objects
  # and account for is_training
  args = parser.args()
  model = Model(args)

  #






if __name__ == "__main__":
  main()


