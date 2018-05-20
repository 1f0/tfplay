import time
import numpy as np
import tensorflow as tf
from load import words, word_to_id, encoded
from model import CharRNN, get_batches

batch_size = 30
n_steps = 20#one to one RNN, affect loss calculation
lstm_size = 1024
n_layers = 2
learning_rate = 0.001
keep_prob = 0.5

eporchs = 5 
report_every_n = 50

mdl = CharRNN(len(words), batch_size=batch_size, n_steps=n_steps,
                lstm_size=lstm_size, n_layers=n_layers,
                learning_rate=learning_rate)

train_logs = './train_logs'
train_summary_writer = tf.summary.FileWriter(train_logs, graph=tf.get_default_graph())
tf.summary.scalar("loss", mdl.loss)
merged_sum = tf.summary.merge_all()

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  counter = 0
  for e in range(eporchs):
    new_state = sess.run(mdl.init_state)
    for x, y in get_batches(encoded, batch_size, n_steps):
      counter += 1
      start = time.time()
      feed = {mdl.inputs: x,
              mdl.targets: y,
              mdl.keep_prob: keep_prob,
              mdl.init_state: new_state}
      batch_loss, new_state, _ = sess.run([mdl.loss,
                                           mdl.final_state,
                                           mdl.optimizer],#why output optimizer here?
                                           feed_dict=feed)
      

      if counter % report_every_n == 0:
        summary = merged_sum.eval(feed_dict=feed)
        train_summary_writer.add_summary(summary, counter)

        end = time.time()
        print('round: {}/{}...'.format(e+1, eporchs),
              'step: {}...'.format(counter),
              'batch loss: {:.4f}...'.format(batch_loss),
              '{:.4f} sec/batch'.format(end-start))
        
    saver.save(sess, 'save/e{}-i{}.ckpt'.format(e, counter))

print("Run 'tensorboard --logdir=./train_logs'. Then open http://0.0.0.0:6006/")
