#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class BaseModel(object):

  @staticmethod
  def conv2d(inputdata, out_channel, kernel_size, padding='SAME', stride=1, w_init=None, b_init=None,
             nl=tf.identity, split=1, use_bias=True, data_format='NHWC', name=None):
    with tf.variable_scope(name):
      in_shape = inputdata.get_shape().as_list()
      channel_axis = 3 if data_format == 'NHWC' else 1
      in_channel = in_shape[channel_axis]
      assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
      assert in_channel % split == 0
      assert out_channel % split == 0

      padding = padding.upper()

      if isinstance(kernel_size, list):
        filter_shape = [kernel_size[0], kernel_size[1]] + \
            [in_channel / split, out_channel]
      else:
        filter_shape = [kernel_size, kernel_size] + \
            [in_channel / split, out_channel]

      if isinstance(stride, list):
        strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' else [
            1, 1, stride[0], stride[1]]
      else:
        strides = [1, stride, stride, 1] if data_format == 'NHWC' else [
            1, 1, stride, stride]

      if w_init is None:
        w_init = tf.contrib.layers.variance_scaling_initializer()
      if b_init is None:
        b_init = tf.constant_initializer()

      w = tf.get_variable('W', filter_shape, initializer=w_init)
      b = None

      if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)

      if split == 1:
        conv = tf.nn.conv2d(inputdata, w, strides, padding,
                            data_format=data_format)
      else:
        inputs = tf.split(inputdata, split, channel_axis)
        kernels = tf.split(w, split, 3)
        outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(outputs, channel_axis)

      ret = nl(tf.nn.bias_add(conv, b, data_format=data_format)
               if use_bias else conv, name=name)

    return ret

  @staticmethod
  def relu(inputdata, name=None):
    return tf.nn.relu(features=inputdata, name=name)

  @staticmethod
  def maxpooling(inputdata, kernel_size, stride=None, padding='VALID', data_format='NHWC', name=None):
    padding = padding.upper()

    if stride is None:
      stride = kernel_size

    if isinstance(kernel_size, list):
      kernel = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else \
          [1, 1, kernel_size[0], kernel_size[1]]
    else:
      kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' else [
          1, 1, kernel_size, kernel_size]

    if isinstance(stride, list):
      strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' else [
          1, 1, stride[0], stride[1]]
    else:
      strides = [1, stride, stride, 1] if data_format == 'NHWC' else [
          1, 1, stride, stride]

    return tf.nn.max_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                          data_format=data_format, name=name)

  @staticmethod
  def dropout(inputdata, keep_prob, noise_shape=None, name=None):
    return tf.nn.dropout(inputdata, keep_prob=keep_prob, noise_shape=noise_shape, name=name)

  @staticmethod
  def fullyconnect(inputdata, out_dim, w_init=None, b_init=None, nl=tf.identity, use_bias=True, name=None):
    shape = inputdata.get_shape().as_list()[1:]
    if None not in shape:
      inputdata = tf.reshape(inputdata, [-1, int(np.prod(shape))])
    else:
      inputdata = tf.reshape(inputdata, tf.stack([tf.shape(inputdata)[0], -1]))

    if w_init is None:
      w_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
      b_init = tf.constant_initializer()

    ret = tf.layers.dense(inputs=inputdata, activation=lambda x: nl(x, name='output'), use_bias=use_bias, name=name,
                          kernel_initializer=w_init, bias_initializer=b_init, trainable=True, units=out_dim)
    return ret

  @staticmethod
  def batch_normalize(inputdata, is_training):
    output = tf.contrib.layers.batch_norm(
        inputdata, scale=True, is_training=is_training, updates_collections=None)
    return output

  @staticmethod
  def squeeze(inputdata, axis=None, name=None):
    return tf.squeeze(input=inputdata, axis=axis, name=name)
