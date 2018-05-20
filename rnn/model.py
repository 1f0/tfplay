import numpy as np
import tensorflow as tf

def get_batches(arr, n_seqs, n_steps):
  batch_size = n_seqs * n_steps
  n_batches = int(len(arr) / batch_size)

  # remove remainder
  arr = arr[:batch_size * n_batches]
  arr = arr.reshape((n_seqs, -1))

  for n in range(0, arr.shape[1], n_steps):
    x = arr[:,n:n+n_steps]
    y = np.zeros_like(x)
    # move forward to loop
    if n_steps == 1:
      y[:, 0] = arr[:, (n+1)%arr.shape[1]]
    else:
      y[:,:-1] = x[:,1:]
      y[:, -1] = y[:, 0]
    yield x, y

def build_inputs(n_seqs, n_steps):
  inputs = tf.placeholder(tf.int32, shape=(n_seqs, n_steps), name='inputs')  
  targets = tf.placeholder(tf.int32, shape=(n_seqs, n_steps), name='targets')
  # keep_prob for dropout
  keep_prob = tf.placeholder(tf.float32, name='keep_prob')
  return inputs, targets, keep_prob

def build_lstm(lstm_size, n_layers, batch_size, keep_prob):
  def lstm_cell():
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    return tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)

  cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(n_layers)])
  init_state = cell.zero_state(batch_size, tf.float32)

  return cell, init_state

def build_output(lstm_output, in_size, out_size):
  seq_out = tf.concat(lstm_output, 1)
  x = tf.reshape(seq_out, [-1, in_size])

  with tf.variable_scope('softmax'):
    softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(out_size))

  # probability distribution
  logits = tf.matmul(x, softmax_w)+softmax_b
  out = tf.nn.softmax(logits, name='predictions')

  return out, logits

def build_loss(logits, targets, lstm_size, n_classes):
  y_one_hot = tf.one_hot(targets, n_classes)
  y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

  loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
  loss = tf.reduce_mean(loss)
  return loss

def build_optimizer(loss, learning_rate, clip_grad):
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), clip_grad)
  train_op = tf.train.AdamOptimizer(learning_rate)
  optimizer = train_op.apply_gradients(zip(grads, tvars))

  return optimizer

class CharRNN:
  def __init__(self, n_classes, batch_size=64, n_steps=50,
                     lstm_size=128, n_layers=2, learning_rate=0.001,
                     clip_grad=5, sampling=False):
    if sampling == True:
      batch_size, n_steps = 1, 1

    tf.reset_default_graph()

    self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, n_steps)

    cell, self.init_state = build_lstm(lstm_size, n_layers, batch_size, self.keep_prob)
    x_one_hot = tf.one_hot(self.inputs, n_classes)

    outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.init_state)
    self.final_state = state

    self.prediction, self.logits = build_output(outputs, lstm_size, n_classes)

    self.loss = build_loss(self.logits, self.targets, lstm_size, n_classes)
    self.optimizer = build_optimizer(self.loss, learning_rate, clip_grad)
