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
import tensorflow as tf
import time

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
    self.initial_state = array_ops.zeros(array_ops.pack([self.batch_size, lstm_layer.state_size]), dtype=tf.float32)
    self.initial_state.set_shape([None, lstm_layer.state_size])
    state = self.initial_state
    with tf.variable_scope("RNN"):
      for time in range(self.batch_len):
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
    loss = sequence_loss_by_example([logits],[tf.reshape(self.targets, [-1])],[tf.ones([self.batch_size*self.batch_len])])
    self.cost = tf.reduce_sum(loss) / self.batch_size / self.batch_len
    self.final_state = state

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self,session,lr_value):
    session.run(tf.assign(self.lr, lr_value))

  def sample(self):
    raise NotImplementedError


def run_epoch(session, model, data, max_batches, args):
  '''
  Run the model under given session for max_batches based on args
  :param model: model on which the operations take place
  :param session: session for tensorflow
  :param data: train, validation or testing data
  :param max_batches: maximum number of batches that can be called
  :param args: arguments provided by user in main
  :return: perplexity
  '''

  # to run a session you need the list of tensors/graph nodes and the feed dict
  # for us its the cost, final_state, and optimizer
  # you feed in the (x,y) pairs, and you also propagate the state across the batches
  state = model.initial_state.eval()
  tot_cost = 0.0
  start_time = time.time()
  iters = 0

  for i in range(max_batches):
    x, y = data.next()
    cur_cost, curr_state, _ = session.run([model.cost,model.final_state,model.train_op],
                feed_dict={model.input_layer: x, model.targets: y, model.initial_state: state})
    tot_cost += cur_cost
    state = curr_state
    iters += args.batch_len

    if i % (max_batches//20) == 0:
      print 'iteration %.3f perplexity: %.3f speed: %.0f wps' %\
            (i, np.exp(tot_cost/iters), iters*args.batch_size/(time.time()-start_time))

  return np.exp(tot_cost/iters)

def main():
  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--filename', type=str, default='text8.zip', help='data location for all data')
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

  args = parser.parse_args()
  url = 'http://mattmahoney.net/dc/'

  # load data
  data = download_data(url, args.filename)
  train, val ,test = train_test_split(data, args.split_ratio)

  batch_train = BatchGenerator(train,args.batch_size,args.batch_len)
  batch_train.create_batches()
  max_batches_train = batch_train.epoch_size

  batch_val = BatchGenerator(val,args.batch_size,args.batch_len)
  batch_val.create_batches()
  max_batches_val = batch_val.epoch_size

  batch_test = BatchGenerator(test,args.batch_size,args.batch_len)
  batch_test.create_batches()
  max_batches_test = batch_test.epoch_size

  # Initialize session and graph
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-0.1,0.1)

    with tf.variable_scope("model",reuse=None,initializer=initializer):
      train_model = Model(args)
    with tf.variable_scope("model",reuse=True,initializer=initializer):
      val_model = Model(args)
      test_model = Model(args)

    tf.initialize_all_variables().run()

    for i in range(args.num_epochs):
      # TODO: Add parameter for max_max_epochs
      lr_decay = args.lr_decay ** max(i-5.0,0.0)
      train_model.assign_lr(session, args.lr_rate*lr_decay)

      # run a complete epoch and return appropriate variables
      train_perplexity = run_epoch(session, train_model, batch_train, max_batches_train, args)
      print 'Epoch %d, Train Perplexity: %.3f' % i+1, train_perplexity

      val_perplexity = run_epoch(session, val_model, batch_val, max_batches_val, args)
      print 'Epoch %d, Val Perplexity: %.3f' % i+1, val_perplexity


    test_perplexity = run_epoch(session, test_model, batch_test, max_batches_test, args)
    print 'Test Perplexity: %.3f' % test_perplexity

if __name__ == "__main__":
  main()


