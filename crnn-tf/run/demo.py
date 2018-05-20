#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os.path as ops
import numpy as np
import cv2
from termcolor import colored

from model.model import CRNN
from run.config import cfg
from run.feaio import FeatureIO
from run.init import init_args


def recognize(image_path, weights_path, is_vis=True):
  decoder = FeatureIO().reader

  image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  image = np.repeat(image[..., None], 3, axis=2)
  gray = image = cv2.resize(image, (100, 32))
  image = np.expand_dims(image, axis=0).astype(np.float32)

  inputdata = tf.placeholder(dtype=tf.float32, shape=[
                             1, 32, 100, 3], name='input')

  net = CRNN(
      phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)

  with tf.variable_scope('crnn'):
    net_out = net.build(inputdata=inputdata)

  decodes, _ = tf.nn.ctc_beam_search_decoder(
      inputs=net_out, sequence_length=25 * np.ones(1), merge_repeated=False)

  # config tf session
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.per_process_gpu_memory_fraction = cfg.TRAIN.GPU_MEMORY_FRACTION
  sess_config.gpu_options.allow_growth = cfg.TRAIN.TF_ALLOW_GROWTH

  # config tf saver
  saver = tf.train.Saver()

  sess = tf.Session(config=sess_config)

  with sess.as_default():

    saver.restore(sess=sess, save_path=weights_path)

    preds = sess.run(decodes, feed_dict={inputdata: image})

    preds = decoder.sparse_tensor_to_str(preds[0])

    print('Predict image {:s} label {:s}'.format(
        ops.split(image_path)[1], colored(preds[0], 'red')))

    if is_vis:
      cv2.imshow(image_path, cv2.resize(gray, (400, 128)))
      cv2.waitKey()

if __name__ == '__main__':
  args = init_args()
  recognize(image_path=args.dataset_dir,
            weights_path=args.weights_path, is_vis=(args.vis is not None))
