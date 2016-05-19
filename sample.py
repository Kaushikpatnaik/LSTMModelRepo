import tensorflow as tf

def sample(sess,model,vocab,init_word='',method='random',num=200):
  '''
  Method to sample the LSTM and generates sentences
  :param sess: tf session for the sampling
  :param model: tf model to be used
  :param vocab: dictionary for converting the characters into internal representations
  :param init_word: starting word for the sampling
  :param method: do you want to generate words based on max probability or random choosen probability
  :param num: number of characters you want to generate
  :return: text generated by the model
  '''

  # the idea is to basically propagate the characters generated into the LSTM state
  # get its output and put them back in again
  state = model.initial_state.eval()
  prediction = tf.zeros([1,len(vocab)])

  if len(init_word) >0:
    print "Seeding with " + str(init_word)
    for c in init_word[:-1]:
      ip = vocab[c]
      [state, prediction] = sess.run([model.final_state, model.output_prob],
                         feed_dict={model.input_layer: ip, model.initial_state: state})
  else:
    print "Seeding with equal probabilities"


def main():
  raise NotImplementedError

if __name__ == '__main__':
  main()