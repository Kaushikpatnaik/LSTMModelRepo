'''
File containing the model definition for the LSTM (only tf.graph())

Dropout needs to accomodated. Thus the model definition should know if training or validation or test
Mutiple layers to be accomodated
Weight tying across time
Loss function
Optimizer
predictions: train, test and validation
saving variables, re-using variables

How I am breaking down the implementation
1) Implementation of basic LSTMcell
   - define using number of hidden states, and bias_offset
   - takes in data, and previous state
   - updates internal variables
   - outputs the hidden unit and state
   - do not worry about reuse of variables etc, just implement for a single time step
2) Implementation of a MultiLSTMcell
   - takes in a cell, and depth variable which specifies how many layers to implement
   - takes in data and previous state
   - updates internal variables
   - returns the final hidden variable, and total cell state
3) Implementation of a dropout layer
4) Building upon the LSTMcell, run it on a sequence (based off of LSTM example)
'''

import tensorflow as tf

class LSTM(object):
  '''A single LSTM unit with one hidden layer'''

  def __init__(self,hidden_layers,offset_bias=1.0):
    '''
    Initialize the LSTM with given number of hidden layers and the offset bias
    :param hidden_layers: number of hidden cells in the LSTM
    :param offset_bias: the bias is usually kept as 1.0 initially for ?? TODO: find out reason
    '''

    self.hidden_layers = hidden_layers
    self.offset_bias = offset_bias
    self.state_size = 2*self.hidden_layers

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
      # Cell state and output are of size batch_size * vocab
      # Input_data is of size batch_size * vocab

      # separate the cell state from output
      c, h = tf.array_ops.split(1,2,state)

      # Overall there are four set of input to hidden weights, and four set of hidden to hidden weights
      # All of them can be processed together as part of one array operation or by creating a function and
      # scoping the results appropriately
      # TODO: Kaushik Add initialization schemes
      def sum_inputs(input_data,h,scope):
        with tf.variable_scope(scope):
          ip2hiddenW = tf.get_variable('ip2hidden',shape=[input_data.shape[1],self.hidden_layers])
          hidden2hiddenW = tf.get_variable('hidden2hidden',shape=[self.hidden_layers,self.hidden_layers])
          biasW = tf.get_variable('biasW',shape=[self.hidden_layers])
          ip2hidden = tf.matmul(input_data,ip2hiddenW) + biasW
          hidden2hidden = tf.matmul(h,hidden2hiddenW)+ biasW
        return ip2hidden + hidden2hidden

      ip_gate = sum_inputs(input_data,h,'input_gate')
      ip_transform = sum_inputs(input_data,h,'input_transform')
      forget_gate = sum_inputs(input_data,h,'forget_gate')
      output_gate = sum_inputs(input_data,h,'output_gate')

      # TODO: Kaushik Add dropout to the final layer
      new_c = c*tf.sigmoid(forget_gate + self.offset_bias) + tf.sigmoid(ip_transform)*tf.tanh(ip_gate)
      new_h = tf.tanh(new_c)*tf.sigmoid(output_gate)

    return new_h, tf.array_ops.concat(1,[new_c,new_h])

class DeepLSTM(object):
  '''A DeepLSTM unit composed of multiple LSTM units'''

  def __init__(self, cells):
    '''
    Create a
    :param cell: list of LSTM cells that are to be stacked
    '''
    self.cells = cells

  def __call__(self, input_data, state, scope=None):
    '''
    Go through multiple layers of the cells and return the final output and all the cell states
    :param input_data: data for the current time step
    :param state: previous cell states for all the layers
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
          curr_state = tf.array_ops.slice(state,[0,curr_pos],[-1,cell.state_size])
          curr_pos += cell.state_size
          # hidden unit is propagated as the input_data
          curr_input, new_state = cell(curr_input,curr_state)
          new_states.append(new_state)
    return curr_input, tf.array_ops.concat(1,new_states)



