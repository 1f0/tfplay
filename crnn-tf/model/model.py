#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers as tflayers
from tensorflow.contrib import rnn

from . import base


class CRNN(base.BaseModel):

  def __init__(self, phase, hidden_nums, layers_nums, seq_length, num_classes):
    super(CRNN, self).__init__()
    self.__phase = phase
    self.__hidden_nums = hidden_nums
    self.__layers_nums = layers_nums
    self.__seq_length = seq_length
    self.__num_classes = num_classes

  @property
  def phase(self):
    return self.__phase

  @phase.setter
  def phase(self, value):
    if not isinstance(value, str):
      raise TypeError('value should be a str \'Test\' or \'Train\'')
    if value.lower() not in ['test', 'train']:
      raise ValueError('value should be a str \'Test\' or \'Train\'')
    self.__phase = value.lower()

  def __conv_stage(self, inputdata, out_dims, name=None):
    conv = self.conv2d(inputdata=inputdata, out_channel=out_dims,
                       kernel_size=3, stride=1, use_bias=False, name=name)
    relu = self.relu(inputdata=conv)
    max_pool = self.maxpooling(inputdata=relu, kernel_size=2, stride=2)
    return max_pool

  def __feature_sequence_extraction(self, inputdata):
    conv1 = self.__conv_stage(
        inputdata=inputdata, out_dims=64, name='conv1')  # batch*16*50*64
    conv2 = self.__conv_stage(
        inputdata=conv1, out_dims=128, name='conv2')  # batch*8*25*128
    conv3 = self.conv2d(inputdata=conv2, out_channel=256, kernel_size=3,
                        stride=1, use_bias=False, name='conv3')  # batch*8*25*256
    relu3 = self.relu(conv3)  # batch*8*25*256
    conv4 = self.conv2d(inputdata=relu3, out_channel=256, kernel_size=3,
                        stride=1, use_bias=False, name='conv4')  # batch*8*25*256
    relu4 = self.relu(conv4)  # batch*8*25*256
    max_pool4 = self.maxpooling(inputdata=relu4, kernel_size=[2, 1], stride=[
                                2, 1], padding='VALID')  # batch*4*25*256
    conv5 = self.conv2d(inputdata=max_pool4, out_channel=512, kernel_size=3,
                        stride=1, use_bias=False, name='conv5')  # batch*4*25*512
    relu5 = self.relu(conv5)  # batch*4*25*512
    if self.phase.lower() == 'train':
      bn5 = self.batch_normalize(inputdata=relu5, is_training=True)
    else:
      bn5 = self.batch_normalize(
          inputdata=relu5, is_training=False)  # batch*4*25*512
    conv6 = self.conv2d(inputdata=bn5, out_channel=512, kernel_size=3,
                        stride=1, use_bias=False, name='conv6')  # batch*4*25*512
    relu6 = self.relu(conv6)  # batch*4*25*512
    if self.phase.lower() == 'train':
      bn6 = self.batch_normalize(inputdata=relu6, is_training=True)
    else:
      bn6 = self.batch_normalize(
          inputdata=relu6, is_training=False)  # batch*4*25*512
    max_pool6 = self.maxpooling(inputdata=bn6, kernel_size=[
                                2, 1], stride=[2, 1])  # batch*2*25*512
    conv7 = self.conv2d(inputdata=max_pool6, out_channel=512, kernel_size=2, stride=[
                        2, 1], use_bias=False, name='conv7')  # batch*1*25*512
    relu7 = self.relu(conv7)  # batch*1*25*512
    return relu7

  def __map_to_sequence(self, inputdata):
    shape = inputdata.get_shape().as_list()
    assert shape[1] == 1  # H of the feature map must equal to 1
    return self.squeeze(inputdata=inputdata, axis=1)

  def __sequence_to_label(self, inputdata):
    with tf.variable_scope('LSTMLayers'):
      # construct stack lstm rcnn layer
      # forward lstm cell
      fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [
          self.__hidden_nums, self.__hidden_nums]]
      # Backward direction cells
      bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [
          self.__hidden_nums, self.__hidden_nums]]

      stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list, inputdata,
                                                                   dtype=tf.float32)

      if self.phase.lower() == 'train':
        stack_lstm_layer = self.dropout(
            inputdata=stack_lstm_layer, keep_prob=0.5)

      [batch_s, _, hidden_nums] = inputdata.get_shape(
      ).as_list()  # [batch, width, 2*n_hidden]

      # [batch x width, 2*n_hidden]
      rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, hidden_nums])

      w = tf.Variable(tf.truncated_normal(
          [hidden_nums, self.__num_classes], stddev=0.1), name="w")
      # Doing the affine projection

      logits = tf.matmul(rnn_reshaped, w)

      logits = tf.reshape(logits, [batch_s, -1, self.__num_classes])

      raw_pred = tf.argmax(tf.nn.softmax(
          logits), axis=2, name='raw_prediction')

      # Swap batch and batch axis
      # [width, batch, n_classes]
      rnn_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')

    return rnn_out, raw_pred

  def build(self, inputdata):
    cnn_out = self.__feature_sequence_extraction(inputdata=inputdata)
    sequence = self.__map_to_sequence(inputdata=cnn_out)
    net_out, raw_pred = self.__sequence_to_label(inputdata=sequence)

    return net_out
