'''
File containing the model definition for the some of the layers I am recreating from tensorflow
for my better understanding
'''

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

class LSTM(object):
  '''A single LSTM unit with one hidden layer'''

  def __init__(self,hidden_units,offset_bias=1.0):
    '''
    Initialize the LSTM with given number of hidden layers and the offset bias
    :param hidden_units: number of hidden cells in the LSTM
    :param offset_bias: the bias is usually kept as 1.0 initially for ?? TODO: find out reason
    '''

    self.hidden_units = hidden_units
    self.offset_bias = offset_bias
    self.state_size = 2*self.hidden_units

  def __call__(self, input_data, state, scope=None):
    '''
    Take in input_data and update the hidden unit and the cell state
    :param input_data: data for the current time step
    :param state: previous cell state
    :param scope: scope within which the variables exist
    :return: new cell state and output concated
    '''
    with tf.variable_scope(scope or type(self).__name__):
      # Recurrent weights are always of size hidden_layers*hidden_layers
      # Input to hidden are always of size vocab_size*hidden_layers
      # Cell state and output are of size batch_size * hidden_units
      # Input_data is of size batch_size * vocab

      # separate the cell state from output
      c, h = array_ops.split(1,2,state)

      # Overall there are four set of input to hidden weights, and four set of hidden to hidden weights
      # All of them can be processed together as part of one array operation or by creating a function and
      # scoping the results appropriately
      # TODO: Kaushik Add initialization schemes
      def sum_inputs(input_data,h,scope):
        with tf.variable_scope(scope):
          ip2hiddenW = tf.get_variable('ip2hidden',shape=[input_data.get_shape()[1],self.hidden_units])
          hidden2hiddenW = tf.get_variable('hidden2hidden',shape=[self.hidden_units,self.hidden_units])
          biasW = tf.get_variable('biasW',shape=[self.hidden_units])
          ip2hidden = tf.matmul(input_data,ip2hiddenW) + biasW
          hidden2hidden = tf.matmul(h,hidden2hiddenW)+ biasW
          return ip2hidden + hidden2hidden

      ip_gate = sum_inputs(input_data,h,'input_gate')
      ip_transform = sum_inputs(input_data,h,'input_transform')
      forget_gate = sum_inputs(input_data,h,'forget_gate')
      output_gate = sum_inputs(input_data,h,'output_gate')

      new_c = c*tf.sigmoid(forget_gate + self.offset_bias) + tf.sigmoid(ip_transform)*tf.tanh(ip_gate)
      new_h = tf.tanh(new_c)*tf.sigmoid(output_gate)

      return new_h, array_ops.concat(1,[new_c,new_h])

  def zero_state(self, batch_size, dtype):
    '''
    return a zero shaped vector (used in initialization schemes)
    :param batch_size: size of batch
    :param dtype: data type of the batch
    :return: a 2D tensor of shape [batch_size x state_size]
    '''
    initial_state = array_ops.zeros(array_ops.pack([batch_size, self.state_size]), dtype=dtype)
    return initial_state

class DeepLSTM(object):
  '''A DeepLSTM unit composed of multiple LSTM units'''

  def __init__(self, cells, drop_prob=0):
    '''
    :param cell: list of LSTM cells that are to be stacked
    :param drop_porb: layerwise regularization using dropout
    '''
    self.cells = cells
    self.state_size = sum([cell.state_size for cell in cells])
    self.drop_prob = drop_prob

  def __call__(self, input_data, state, is_training, scope=None):
    '''
    Go through multiple layers of the cells and return the final output and all the cell states
    :param input_data: data for the current time step
    :param state: previous cell states for all the layers
    :param is_training: boolean flag capturing whether training is being done or not
    :param scope: scope within which the operation will occur
    :return: new cell states and final output layer
    '''
    with tf.variable_scope(scope or type(self).__name__):
      # with multiple layers we need to iterate through each layer, and update its weights and cell states
      # to ensure no collision among weights, we should scope within the layer loop also
      new_states = []
      curr_pos = 0
      curr_input = input_data
      for i,cell in enumerate(self.cells):
        with tf.variable_scope("Cell_"+str(i)):
          curr_state = array_ops.slice(state,[0,curr_pos],[-1,cell.state_size])
          curr_pos += cell.state_size
          # hidden unit is propagated as the input_data
          curr_input, new_state = cell(curr_input,curr_state)
          if self.drop_prob and is_training:
            curr_input = dropout(curr_input,self.drop_prob)
          new_states.append(new_state)
      return curr_input, array_ops.concat(1,new_states)

  def zero_state(self, batch_size, dtype):
    '''
    return a zero shaped vector (used in initialization schemes)
    :param batch_size: size of batch
    :param dtype: data type of the batch
    :return: a 2D tensor of shape [batch_size x state_size]
    '''
    initial_state = array_ops.zeros(array_ops.pack([batch_size, self.state_size]), dtype=dtype)
    return initial_state

# TODO: Kaushik handle sequences of different lengths, and accomodate dropout
def fixed_time_steps_LSTM(cell, inputs, initial_state = None, scope = None):
  '''Run a LSTM or DeepLSTM for multiple time steps (prefixed)'''

  with tf.variable_scope(scope or "RNN"):
    if initial_state is not None:
      state = initial_state

    outputs = []
    for time, input in enumerate(inputs):
      if time > 0: tf.get_variable_scope().reuse_variables()
      output, state = cell(input, state)
      outputs.append(output)

    return (outputs, state)

def dropout(x, dropout_prob, seed=None, name=None):

  with tf.variable_scope(name or 'Dropout'):
    if isinstance(dropout,float) and not 0<dropout_prob<=1:
      raise ValueError("dropout probability must be a scalar tensor or a value in "
                       "range (0,1]")
    x = tf.convert_to_tensor(x)
    dropout_prob = tf.convert_to_tensor(dropout_prob,dtype=x.dtype)
    random_tensor = tf.random_uniform(x.get_shape(),minval=0,maxval=1,dtype=x.dtype,seed=seed)
    binary_tensor = tf.floor(random_tensor+dropout_prob)
    ret = x * tf.inv(dropout_prob) * binary_tensor
    ret.set_shape(x.get_shape())
    return ret

# TODO: DeepLSTM with recurrent batch normalization





