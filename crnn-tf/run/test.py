#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import os.path as ops
import tensorflow as tf
import argparse
import numpy as np
from math import ceil

from model.model import CRNN
from run.config import cfg
from run.feaio import FeatureIO
from run.init import init_args


def test(dataset_dir, weights_path):
  # Initialize the record decoder
  decoder = FeatureIO().reader
  tfrec_path = ops.join(dataset_dir, 'test.tfrecords')
  images, labels, imagenames = decoder.read_features(
      tfrec_path, num_epochs=None)
  inputdata, input_labels, input_imagenames = tf.train.batch(tensors=[images, labels, imagenames],
                                                             batch_size=32, capacity=1000 + 32 * 2, num_threads=4)

  inputdata = tf.cast(x=inputdata, dtype=tf.float32)

  net = CRNN(
      phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)

  with tf.variable_scope('crnn'):
    net_out = net.build(inputdata=inputdata)

  decoded, _ = tf.nn.ctc_beam_search_decoder(
      net_out, 25 * np.ones(32), merge_repeated=False)

  # config tf session
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.per_process_gpu_memory_fraction = cfg.TRAIN.GPU_MEMORY_FRACTION
  sess_config.gpu_options.allow_growth = cfg.TRAIN.TF_ALLOW_GROWTH

  # config tf saver
  saver = tf.train.Saver()

  sess = tf.Session(config=sess_config)

  test_sample_count = 0
  for record in tf.python_io.tf_record_iterator(tfrec_path):
    test_sample_count += 1
  loops_nums = int(ceil(test_sample_count / 32))

  with sess.as_default():

    # restore the model weights
    saver.restore(sess=sess, save_path=weights_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('Start predicting ......')
    accuracy = []
    for epoch in range(loops_nums):
      predictions, images, labels, imagenames = sess.run(
          [decoded, inputdata, input_labels, input_imagenames])
      imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
      imagenames = [tmp.decode('utf-8') for tmp in imagenames]
      preds_res = decoder.sparse_tensor_to_str(predictions[0])
      gt_res = decoder.sparse_tensor_to_str(labels)

      for index, gt_label in enumerate(gt_res):
        pred = preds_res[index]
        totol_count = len(gt_label)
        correct_count = 0
        try:
          for i, tmp in enumerate(gt_label):
            if tmp == pred[i]:
              correct_count += 1
        except IndexError:
          continue
        finally:
          try:
            accuracy.append(correct_count / totol_count)
          except ZeroDivisionError:
            if len(pred) == 0:
              accuracy.append(1)
            else:
              accuracy.append(0)

      for index, image in enumerate(images):
        print('{:s} with gt label: {:s} | predict label: {:s}'.format(
            imagenames[index], gt_res[index], preds_res[index]))

    accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    print('Test accuracy is {:5f}, with {:d} samples.'.format(accuracy, test_sample_count))

    coord.request_stop()
    coord.join(threads=threads)

  sess.close()
  return


if __name__ == '__main__':
  args = init_args()
  test(args.dataset_dir, args.weights_path)
