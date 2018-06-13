#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2017 Timothy Dozat
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from . import nn
from . import nonlin

#***************************************************************
def hidden(layer, hidden_size, hidden_func=nonlin.relu, hidden_keep_prob=1.):
  """"""
  
  layer_shape = nn.get_sizes(layer)
  input_size = layer_shape.pop()
  weights = tf.get_variable('Weights', shape=[input_size, hidden_size], initializer=tf.orthogonal_initializer)
  biases = tf.get_variable('Biases', shape=[hidden_size], initializer=tf.zeros_initializer)
  if hidden_keep_prob < 1.:
    if len(layer_shape) > 1:
      noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    else:
      noise_shape = None
    layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
  
  layer = nn.reshape(layer, [-1, input_size])
  layer = tf.matmul(layer, weights) + biases
  layer = hidden_func(layer)
  layer = nn.reshape(layer, layer_shape + [hidden_size])
  return layer

#===============================================================
def hiddens(layer, hidden_sizes, hidden_func=nonlin.relu, hidden_keep_prob=1.):
  """"""
  
  layer_shape = nn.get_sizes(layer)
  input_size = layer_shape.pop()
  weights = []
  for i, hidden_size in enumerate(hidden_sizes):
    weights.append(tf.get_variable('Weights-%d' % i, shape=[input_size, hidden_size], initializer=tf.orthogonal_initializer))
  weights = tf.concat(weights, axis=1)
  hidden_size = sum(hidden_sizes)
  biases = tf.get_variable('Biases', shape=[hidden_size], initializer=tf.zeros_initializer)
  if hidden_keep_prob < 1.:
    if len(layer_shape) > 1:
      noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    else:
      noise_shape = None
    layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
  
  layer = nn.reshape(layer, [-1, input_size])
  layer = tf.matmul(layer, weights) + biases
  layer = hidden_func(layer)
  layer = nn.reshape(layer, layer_shape + [hidden_size])
  layers = tf.split(layer, hidden_sizes, axis=-1)
  return layers

#===============================================================
def linear_classifier(layer, output_size, hidden_keep_prob=1.):
  """"""
  
  layer_shape = nn.get_sizes(layer)
  input_size = layer_shape.pop()
  weights = tf.get_variable('Weights', shape=[input_size, output_size], initializer=tf.zeros_initializer)
  biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
  if hidden_keep_prob < 1.:
    if len(layer_shape) > 1:
      noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    else:
      noise_shape = None
    layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
  
  # (n x m x d) -> (nm x d)
  layer_reshaped = nn.reshape(layer, [-1, input_size])
  
  # (nm x d) * (d x o) -> (nm x o)
  layer = tf.matmul(layer_reshaped, weights) + biases
  # (nm x o) -> (n x m x o)
  layer = nn.reshape(layer, layer_shape + [output_size])
  return layer
  
#===============================================================
def linear_attention(layer, hidden_keep_prob=1.):
  """"""
  
  layer_shape = nn.get_sizes(layer)
  input_size = layer_shape.pop()
  weights = tf.get_variable('Weights', shape=[input_size, 1], initializer=tf.zeros_initializer)
  if hidden_keep_prob < 1.:
    if len(layer_shape) > 1:
      noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    else:
      noise_shape = None
    layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
  
  # (n x m x d) -> (nm x d)
  layer_reshaped = tf.reshape(layer, [-1, input_size])
  
  # (nm x d) * (d x 1) -> (nm x 1)
  attn = tf.matmul(layer_reshaped, weights)
  # (nm x 1) -> (n x m)
  attn = tf.reshape(attn, layer_shape)
  # (n x m) -> (n x m)
  soft_attn = tf.nn.sigmoid(attn)
  # (n x m) -> (n x 1 x m)
  soft_attn = tf.expand_dims(soft_attn, axis=-2)
  # (n x 1 x m) * (n x m x d) -> (n x 1 x d)
  weighted_layer = tf.matmul(soft_attn, layer)
  # (n x 1 x d) -> (n x d)
  weighted_layer = tf.squeeze(weighted_layer, -2)
  return attn, weighted_layer
  
#===============================================================
def bilinear_classifier(layer1, layer2, output_size, hidden_keep_prob=1., add_linear=True):
  """"""
  
  layer_shape = nn.get_sizes(layer1)
  bucket_size = layer_shape[-2]
  input1_size = layer_shape.pop()+add_linear
  input2_size = layer2.get_shape().as_list()[-1]+add_linear
  ones_shape = tf.stack(layer_shape + [1])
  
  weights = tf.get_variable('Weights', shape=[input1_size, output_size, input2_size], initializer=tf.zeros_initializer)
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
  if hidden_keep_prob < 1.:
    noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-add_linear])
    noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-add_linear])
    layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
    layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
  if add_linear:
    ones = tf.ones(ones_shape)
    layer1 = tf.concat([layer1, ones], -1)
    layer2 = tf.concat([layer2, ones], -1)
    biases = 0
  else:
    biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
    # (o) -> (o x 1)
    biases = nn.reshape(biases, [output_size, 1])
  
  # (n x m x d) -> (nm x d)
  layer1 = nn.reshape(layer1, [-1, input1_size])
  # (n x m x d) -> (n x m x d)
  layer2 = nn.reshape(layer2, [-1, bucket_size, input2_size])
  # (d x o x d) -> (d x od)
  weights = nn.reshape(weights, [input1_size, output_size*input2_size])
  
  # (nm x d) * (d x od) -> (nm x od)
  layer = tf.matmul(layer1, weights)
  # (nm x od) -> (n x mo x d)
  layer = nn.reshape(layer, [-1, bucket_size*output_size, input2_size])
  # (n x mo x d) * (n x m x d) -> (n x mo x m)
  layer = tf.matmul(layer, layer2, transpose_b=True)
  # (n x mo x m) -> (n x m x o x m)
  layer = nn.reshape(layer, layer_shape + [output_size, bucket_size]) + biases
  return layer

#===============================================================
def diagonal_bilinear_classifier(layer1, layer2, output_size, hidden_keep_prob=1., add_linear=True):
  """"""
  
  layer_shape = nn.get_sizes(layer1)
  bucket_size = layer_shape[-2]
  input1_size = layer_shape.pop()
  input2_size = layer2.get_shape().as_list()[-1]
  assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
  input_size = input1_size
  ones_shape = tf.stack(layer_shape + [1])
  
  weights = tf.get_variable('Weights', shape=[input_size, output_size], initializer=tf.zeros_initializer)
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
  if add_linear:
    weights1 = tf.get_variable('Weights1', shape=[input_size, output_size], initializer=tf.zeros_initializer)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
    weights2 = tf.get_variable('Weights2', shape=[input_size, output_size], initializer=tf.zeros_initializer)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
  biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
  if hidden_keep_prob < 1.:
    noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
    layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
  
  if add_linear:
    # (n x m x d) -> (nm x d)
    lin_layer1 = nn.reshape(layer1, [-1, input_size])
    # (nm x d) * (d x o) -> (nm x o)
    lin_layer1 = tf.matmul(lin_layer1, weights1)
    # (nm x o) -> (n x m x o)
    lin_layer1 = nn.reshape(lin_layer1, layer_shape + [output_size])
    # (n x m x o) -> (n x m x o x 1)
    lin_layer1 = tf.expand_dims(lin_layer1, axis=-1)
    # (n x m x d) -> (nm x d)
    lin_layer2 = nn.reshape(layer2, [-1, input_size])
    # (nm x d) * (d x o) -> (nm x o)
    lin_layer2 = tf.matmul(lin_layer2, weights1)
    # (nm x o) -> (n x m x o)
    lin_layer2 = nn.reshape(lin_layer2, layer_shape + [output_size])
    # (n x m x o) -> (n x o x m)
    lin_layer2 = tf.transpose(lin_layer2, [0, 2, 1])
    # (n x o x m) -> (n x 1 x o x m)
    lin_layer2 = tf.expand_dims(lin_layer2, axis=-3)
  
  # (n x m x d) -> (n x m x 1 x d)
  layer1 = nn.reshape(layer1, [-1, bucket_size, 1, input_size])
  # (n x m x d) -> (n x m x d)
  layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
  # (d x o) -> (o x d)
  weights = tf.transpose(weights, [1, 0])
  # (o) -> (o x 1)
  biases = nn.reshape(biases, [output_size, 1])
  
  # (n x m x 1 x d) (*) (o x d) -> (n x m x o x d)
  layer = layer1 * weights
  # (n x m x o x d) -> (n x mo x d)
  layer = nn.reshape(layer, [-1, bucket_size*output_size, input_size])
  # (n x mo x d) * (n x m x d) -> (n x mo x m)
  layer = tf.matmul(layer, layer2, transpose_b=True)
  # (n x mo x m) -> (n x m x o x m)
  layer = nn.reshape(layer, layer_shape + [output_size, bucket_size])
  if add_linear:
    # (n x m x o x m) + (n x 1 x o x m) + (n x m x o x 1) -> (n x m x o x m)
    layer += lin_layer1 + lin_layer2
  # (n x m x o x m) + (o x 1) -> (n x m x o x m)
  layer += biases
  return layer

#===============================================================
def bilinear_discriminator(layer1, layer2, hidden_keep_prob=1., add_linear=True):
  """"""
  
  layer_shape = nn.get_sizes(layer1)
  bucket_size = layer_shape[-2]
  input1_size = layer_shape.pop()+1
  input2_size = layer2.get_shape().as_list()[-1]+1
  ones_shape = tf.stack(layer_shape + [1])
  
  weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.zeros_initializer)
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
  if hidden_keep_prob < 1.:
    noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
    noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
    layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
    layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
  ones = tf.ones(ones_shape)
  layer1 = tf.concat([layer1, ones], -1)
  layer2 = tf.concat([layer2, ones], -1)
  
  # (n x m x d) -> (nm x d)
  layer1 = nn.reshape(layer1, [-1, input1_size])
  # (n x m x d) -> (n x m x d)
  layer2 = tf.reshape(layer2, [-1, bucket_size, input2_size])
  
  # (nm x d) * (d x d) -> (nm x d)
  layer = tf.matmul(layer1, weights)
  # (nm x d) -> (n x m x d)
  layer = nn.reshape(layer, [-1, bucket_size, input2_size])
  # (n x m x d) * (n x m x d) -> (n x m x m)
  layer = tf.matmul(layer, layer2, transpose_b=True)
  # (n x mo x m) -> (n x m x m)
  layer = nn.reshape(layer, layer_shape + [bucket_size])
  return layer

#===============================================================
def diagonal_bilinear_discriminator(layer1, layer2, hidden_keep_prob=1., add_linear=True):
  """"""
  
  layer_shape = nn.get_sizes(layer1)
  bucket_size = layer_shape[-2]
  input1_size = layer_shape.pop()
  input2_size = layer2.get_shape().as_list()[-1]
  assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
  input_size = input1_size
  ones_shape = tf.stack(layer_shape + [1])
  
  weights = tf.get_variable('Weights', shape=[input_size], initializer=tf.zeros_initializer)
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
  if add_linear:
    weights1 = tf.get_variable('Weights1', shape=[input_size], initializer=tf.zeros_initializer)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
    weights2 = tf.get_variable('Weights2', shape=[input_size], initializer=tf.zeros_initializer)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
  biases = tf.get_variable('Biases', shape=[1], initializer=tf.zeros_initializer)
  if hidden_keep_prob < 1.:
    noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
    layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
  
  if add_linear:
    #(d) -> (d x 1)
    weights1 = tf.expand_dims(weights1, axis=-1)
    # (n x m x d) -> (nm x d)
    lin_layer1 = nn.reshape(layer1, [-1, input_size])
    # (nm x d) * (d x 1) -> (nm x 1)
    lin_layer1 = tf.matmul(lin_layer1, weights1)
    # (nm x 1) -> (n x m)
    lin_layer1 = nn.reshape(lin_layer1, layer_shape)
    # (n x m) -> (n x m x 1)
    lin_layer1 = tf.expand_dims(lin_layer1, axis=-1)
    #(d) -> (d x 1)
    weights2 = tf.expand_dims(weights2, axis=-1)
    # (n x m x d) -> (nm x d)
    lin_layer2 = nn.reshape(layer2, [-1, input_size])
    # (nm x d) * (d x 1) -> (nm x 1)
    lin_layer2 = tf.matmul(lin_layer2, weights1)
    # (nm x 1) -> (n x m)
    lin_layer2 = nn.reshape(lin_layer2, layer_shape)
    # (n x m) -> (n x 1 x m)
    lin_layer2 = tf.expand_dims(lin_layer2, axis=-2)
  
  # (n x m x d) -> (n x m x d)
  layer1 = nn.reshape(layer1, [-1, bucket_size, input_size])
  # (n x m x d) -> (n x m x d)
  layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
  
  # (n x m x d) (*) (d) -> (n x m x d)
  layer = layer1 * weights
  # (n x m x d) * (n x m x d) -> (n x m x m)
  layer = tf.matmul(layer, layer2, transpose_b=True)
  # (n x m x m) -> (n x m x m)
  layer = nn.reshape(layer, layer_shape + [bucket_size])
  if add_linear:
    # (n x m x m) + (n x 1 x m) + (n x m x 1) -> (n x m x m)
    layer += lin_layer1 + lin_layer2
  # (n x m x m) + () -> (n x m x m)
  layer += biases
  return layer

#===============================================================
def bilinear_attention(layer1, layer2, hidden_keep_prob=1., add_linear=True):
  """"""
  
  layer_shape = nn.get_sizes(layer1)
  bucket_size = layer_shape[-2]
  input1_size = layer_shape.pop()+add_linear
  input2_size = layer2.get_shape().as_list()[-1]
  ones_shape = tf.stack(layer_shape + [1])
  
  weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.zeros_initializer)
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
  original_layer1 = layer1
  if hidden_keep_prob < 1.:
    noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-add_linear])
    noise_shape2 = tf.stack(layer_shape[:-2] + [1, input2_size])
    layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
    layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
  if add_linear:
    ones = tf.ones(ones_shape)
    layer1 = tf.concat([layer1, ones], -1)
  
  # (n x m x d) -> (nm x d)
  layer1 = nn.reshape(layer1, [-1, input1_size])
  # (n x m x d) -> (n x m x d)
  layer2 = nn.reshape(layer2, [-1, bucket_size, input2_size])
  
  # (nm x d) * (d x d) -> (nm x d)
  attn = tf.matmul(layer1, weights)
  # (nm x d) -> (n x m x d)
  attn = nn.reshape(attn, [-1, bucket_size, input2_size])
  # (n x m x d) * (n x m x d) -> (n x m x m)
  attn = tf.matmul(attn, layer2, transpose_b=True)
  # (n x m x m) -> (n x m x m)
  attn = nn.reshape(attn, layer_shape + [bucket_size])
  # (n x m x m) -> (n x m x m)
  soft_attn = tf.nn.softmax(attn)
  # (n x m x m) * (n x m x d) -> (n x m x d)
  weighted_layer1 = tf.matmul(soft_attn, original_layer1)
  
  return attn, weighted_layer1
  
#===============================================================
def diagonal_bilinear_attention(layer1, layer2, hidden_keep_prob=1., add_linear=True):
  """"""
  
  layer_shape = nn.get_sizes(layer1)
  bucket_size = layer_shape[-2]
  input1_size = layer_shape.pop()
  input2_size = layer2.get_shape().as_list()[-1]
  assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
  input_size = input1_size
  ones_shape = tf.stack(layer_shape + [1])
  
  weights = tf.get_variable('Weights', shape=[input_size], initializer=tf.zeros_initializer)
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
  if add_linear:
    weights2 = tf.get_variable('Weights2', shape=[input_size], initializer=tf.zeros_initializer)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
  original_layer1 = layer1
  if hidden_keep_prob < 1.:
    noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
    layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
  
  if add_linear:
    #(d) -> (d x 1)
    weights2 = tf.expand_dims(weights2, axis=-1)
    # (n x m x d) -> (nm x d)
    lin_attn2 = nn.reshape(layer2, [-1, input_size])
    # (nm x d) * (d x 1) -> (nm x 1)
    lin_attn2 = tf.matmul(lin_attn2, weights2)
    # (nm x 1) -> (n x m)
    lin_attn2 = nn.reshape(lin_attn2, layer_shape)
    # (n x m) -> (n x 1 x m)
    lin_attn2 = tf.expand_dims(lin_attn2, axis=-2)
  
  # (n x m x d) -> (nm x d)
  layer1 = nn.reshape(layer1, [-1, input_size])
  # (n x m x d) -> (n x m x d)
  layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
  
  # (nm x d) * (d) -> (nm x d)
  attn = layer1 * weights
  # (nm x d) -> (n x m x d)
  attn = nn.reshape(attn, [-1, bucket_size, input_size])
  # (n x m x d) * (n x m x d) -> (n x m x m)
  attn = tf.matmul(attn, layer2, transpose_b=True)
  # (n x m x m) -> (n x m x m)
  attn = nn.reshape(attn, layer_shape + [bucket_size])
  if add_linear:
    # (n x m x m) + (n x 1 x m) -> (n x m x m)
    attn += lin_attn2
  # (n x m x m) -> (n x m x m)
  soft_attn = tf.nn.softmax(attn)
  # (n x m x m) * (n x m x d) -> (n x m x d)
  weighted_layer1 = tf.matmul(soft_attn, original_layer1)

  return attn, weighted_layer1
