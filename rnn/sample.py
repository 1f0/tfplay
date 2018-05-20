# -*- coding: utf-8 -*-

from __future__ import print_function
from model import CharRNN
from load import words, word_to_id, id_to_word
import numpy as np
import tensorflow as tf
import sys

def pick_top_n(prediction, word_size, top_n=5):
  p = np.squeeze(prediction)
  p[np.argsort(p)[:-top_n]] = 0
  p = p / np.sum(p)# normalize
  # choose character randomly
  c = np.random.choice(word_size, 1, p=p)[0]
  return c

def sample(checkpoint, n_samples, lstm_size, word_size, prime="The "):
  samples = [c for c in prime]
  mdl = CharRNN(len(words), lstm_size=lstm_size, sampling=True)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    saver.restore(sess, checkpoint)
    new_state = sess.run(mdl.init_state)
    for c in prime:
        x = np.zeros((1, 1))
        x[0,0] = word_to_id[c]
        feed = {mdl.inputs: x,
                mdl.keep_prob: 1.,
                mdl.init_state: new_state}
        preds, new_state = sess.run([mdl.prediction, mdl.final_state], 
                                     feed_dict=feed)

    c = pick_top_n(preds, len(words))
    # add character to samples
    samples.append(id_to_word[c])
    
    # generate characters
    for i in range(n_samples):
        x[0,0] = c
        feed = {mdl.inputs: x,
                mdl.keep_prob: 1.,
                mdl.init_state: new_state}
        preds, new_state = sess.run([mdl.prediction, mdl.final_state], 
                                     feed_dict=feed)

        c = pick_top_n(preds, len(words))
        samples.append(id_to_word[c])

  return ''.join(samples)

lstm_size = 1024 # this should be kept with train.py
checkpoint = tf.train.latest_checkpoint('save')
#checkpoint = 'save/e0-i15824.ckpt' 
samp = sample(checkpoint, 800, lstm_size, len(words), u"宝玉")
print(samp)

if len(sys.argv) > 2:
  with open(sys.argv[2], 'w') as f:
    f.write(samp.encode('utf-8'))
