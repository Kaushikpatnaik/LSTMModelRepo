'''
Training file with functions for

1) Taking in the inputs
2) Defining the model
3) Reading the input and generating batches
4) Defining the loss, learning rate and optimization functions
5) Running multiple epochs on training and testing
'''
import argparse
from read_input import *
from model import *
import tensorflow as tf
import time

def run_epoch(session, model, train_op, data, max_batches, args):
  '''
  Run the model under given session for max_batches based on args
  :param model: model on which the operations take place
  :param session: session for tensorflow
  :param train_op: training output variable name, pass as tf.no_op() for validation and testing
  :param data: train, validation or testing data
  :param max_batches: maximum number of batches that can be called
  :param args: arguments provided by user in main
  :return: perplexity
  '''

  # to run a session you need the list of tensors/graph nodes and the feed dict
  # for us its the cost, final_state, and optimizer
  # you feed in the (x,y) pairs, and you also propagate the state across the batches
  state = np.zeros((args.batch_size,model.lstm_layer.state_size))
  tot_cost = 0.0
  start_time = time.time()
  iters = 0

  for i in range(max_batches):
    x, y = data.next()
    cur_cost, curr_state, _ = session.run([model.cost,model.final_state,train_op],
                feed_dict={model.input_layer: x, model.targets: y, model.initial_state: state})
    tot_cost += cur_cost
    state = curr_state
    iters += args.batch_len

    if i % (max_batches//50) == 0:
      print 'iteration %.3f perplexity: %.3f speed: %.0f wps' %\
            (i, np.exp(tot_cost/iters), iters*args.batch_size/(time.time()-start_time))

  return np.exp(tot_cost/iters)

# TODO: Add model saving and loading
def main():
  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--filename', type=str, default='./data/tinyshakespeare/input.txt', help='data location for all data')
  parser.add_argument('--split_ratio', type =list, default=[0.9,0.05,0.05], help='split ratio for train, validation and test')
  parser.add_argument('--batch_size', type=int, default=1, help='batch size for data')
  parser.add_argument('--batch_len', type=int, default=1, help='number of time steps to unroll')
  parser.add_argument('--cell', type=str, default='lstm', help='the cell type to use, currently only LSTM')
  parser.add_argument('--num_layers', type=int, default=1, help='depth of hidden units in the model')
  parser.add_argument('--hidden_units', type=int, default=32, help='number of hidden units in the cell')
  parser.add_argument('--num_epochs', type=int, default=50, help='max number of epochs to run the training')
  parser.add_argument('--lr_rate', type=float, default=2e-5, help='learning rate')
  parser.add_argument('--lr_decay', type=float, default=0.97, help='learning rate decay')
  parser.add_argument('--drop_prob', type=float, default=0, help='optimization function to be used')
  parser.add_argument('--grad_clip', type=float, default=5.0, help='clip gradients at this value')
  parser.add_argument('--stateful', type=bool, default=True, help='save at every batches')

  args = parser.parse_args()

  # load data
  if args.filename[-3:] == 'zip':
    data = load_zip_data(args.filename)
  elif args.filename[-3:] == 'txt':
    data = load_csv_file(args.filename)
  else:
    raise NotImplementedError("File extension not supported")

  train, val ,test = train_test_split(data, args.split_ratio)

  batch_train = BatchGenerator(train,args.batch_size,args.batch_len)
  batch_train.create_batches()
  max_batches_train = batch_train.epoch_size
  # New chars seen in test time will have a problem
  args.data_dim = batch_train.vocab_size

  batch_val = BatchGenerator(val,args.batch_size,args.batch_len)
  batch_val.create_batches()
  max_batches_val = batch_val.epoch_size

  batch_test = BatchGenerator(test,args.batch_size,args.batch_len)
  batch_test.create_batches()
  max_batches_test = batch_test.epoch_size

  print max_batches_train, max_batches_val, max_batches_test

  # Initialize session and graph
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-0.1,0.1)

    with tf.variable_scope("model",reuse=None,initializer=initializer):
      train_model = Model(args, is_training=True, is_inference=False)
    with tf.variable_scope("model",reuse=True,initializer=initializer):
      val_model = Model(args, is_training=False, is_inference=False)
      test_model = Model(args, is_training=False, is_inference=False)

    tf.initialize_all_variables().run()

    for i in range(args.num_epochs):
      # TODO: Add parameter for max_max_epochs
      lr_decay = args.lr_decay ** max(i-10.0,0.0)
      train_model.assign_lr(session, args.lr_rate*lr_decay)

      # run a complete epoch and return appropriate variables
      train_perplexity = run_epoch(session, train_model, train_model.train_op, batch_train, max_batches_train, args)
      print 'Epoch %d, Train Perplexity: %.3f' %(i+1, train_perplexity)

      val_perplexity = run_epoch(session, val_model, tf.no_op(), batch_val, max_batches_val, args)
      print 'Epoch %d, Val Perplexity: %.3f' %(i+1, val_perplexity)

    test_perplexity = run_epoch(session, test_model, tf.no_op(), batch_test, max_batches_test, args)
    print 'Test Perplexity: %.3f' % test_perplexity

if __name__ == "__main__":
  main()


