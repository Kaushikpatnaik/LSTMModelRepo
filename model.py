from layers import *
from tensorflow.python.ops import math_ops
import tensorflow as tf

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
  '''Class defining the overall model based on layers.py'''
  # TODO: passing multiple flags to the model initiliazation seems hacky, better design or maybe a feedforward function
  def __init__(self, args, is_training, is_inference):
    self.batch_size = args.batch_size
    self.batch_len = args.batch_len
    if is_inference:
      self.batch_size = 1
      self.batch_len = 1
    self.num_layers = args.num_layers
    self.cell = args.cell
    self.hidden_units = args.hidden_units
    self.data_dim = args.data_dim
    self.drop_prob = args.drop_prob
    self.is_training = is_training
    self.is_inference = is_inference

    # define placeholder for data layer
    self.input_layer = tf.placeholder(tf.int32, [self.batch_size, self.batch_len])

    # define weights for data layer (vocab_size to self.hidden_units)
    input_weights = tf.get_variable('input_weights', [self.data_dim, self.hidden_units])

    # define input to LSTM cell [self.batch_size x self.batch_len x self.hidden_units]
    inputs = tf.nn.embedding_lookup(input_weights, self.input_layer)

    # define model based on cell and num_layers
    if self.num_layers ==1:
      self.lstm_layer = LSTM(self.hidden_units,self.drop_prob)
    else:
      cells = [LSTM(self.hidden_units,self.drop_prob)]*self.num_layers
      self.lstm_layer = DeepLSTM(cells)

    outputs = []
    # keep the initial_state accessible (as this will be used for initialization) and state resets with epochs
    #self.initial_state = self.lstm_layer.zero_state(self.batch_size,tf.float32)
    self.initial_state = tf.placeholder(tf.float32,[self.batch_size, self.lstm_layer.state_size])
    state = self.initial_state
    # run the model for multiple time steps
    with tf.variable_scope("RNN"):
      for time in range(self.batch_len):
        if time > 0: tf.get_variable_scope().reuse_variables()
        # pass the inputs, state and weather we are in train/test or inference time (for dropout)
        output, state = self.lstm_layer(inputs[:,time,:], state, (self.is_training or self.is_inference))
        outputs.append(output)

    # for each single input collect the hidden units, then reshape as [self.batch_len x self.batch_size x self.hidden_units]
    output = tf.reshape(tf.concat(1,outputs), [-1,self.hidden_units])
    softmax_w = tf.get_variable('softmax_w', [self.hidden_units, self.data_dim])
    softmax_b = tf.get_variable('softmax_b', [self.data_dim])

    # logits is now of shape [self.batch_size x self.batch_len, self.data_dim]
    self.logits = tf.matmul(output, softmax_w) + softmax_b

    # get probabilities for these logits through softmax (will be needed for sampling)
    self.output_prob = tf.nn.softmax(self.logits)

    # define placeholder for target layer
    self.targets = tf.placeholder(tf.int32, [self.batch_size, self.batch_len])

    # sequence loss by example
    # to enable comparision by each and every example the row lengths of logits
    # and targets should be same
    loss = sequence_loss_by_example([self.logits],[tf.reshape(self.targets, [-1])],[tf.ones([self.batch_size*self.batch_len])])
    self.cost = tf.reduce_sum(loss) / self.batch_size / self.batch_len
    self.final_state = state

    if not self.is_training and not self.is_inference:
      return

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    # clips all gradients, including the weight vectors
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self,session,lr_value):
    session.run(tf.assign(self.lr, lr_value))










