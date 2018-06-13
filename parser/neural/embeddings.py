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
from .classifiers import linear_attention

#***************************************************************
def dropout(layer, embed_keep_prob):
  """"""
  
  # Get the dimensions
  batch_size, bucket_size, input_size = nn.get_sizes(layer)
  
  # Set up the mask 
  mask_shape = tf.stack([batch_size, bucket_size, 1])
  mask = nn.binary_mask(mask_shape, embed_keep_prob)
  
  layer = mask*layer
  
  return layer

#===============================================================
def normal_randout(layer, embed_keep_prob):
  """"""
  
  # Get the dimensions
  batch_size, bucket_size, input_size = nn.get_sizes(layer)
  
  # Set up the mask 
  mask_shape = tf.stack([batch_size, bucket_size, 1])
  mask = nn.binary_mask(mask_shape, embed_keep_prob)
  
  # Replace with random vector
  var = tf.nn.moments(layer, [0,1,2])[1]+1e-12
  stddev = tf.sqrt(var)
  rand = tf.random_normal(mask_shape, stddev=stddev)
  layer = mask*layer + (1-mask)*rand
  
  return layer

#===============================================================
def unkout(layer, embed_keep_prob):
  """"""
  
  # Get the dimensions
  batch_size, bucket_size, input_size = nn.get_sizes(layer)
  
  # Set up the mask 
  mask_shape = tf.stack([batch_size, bucket_size, 1])
  mask = nn.binary_mask(mask_shape, embed_keep_prob)
  
  # Get the unk vector
  unk = tf.get_variable('Unk', input_size)
  layer = mask * layer + (1-mask) * unk
  
  return layer
  
#===============================================================
def uniform_randout(layer, embed_keep_prob):
  """"""
  
  # Get the dimensions
  batch_size, bucket_size, input_size = nn.get_sizes(layer)
  
  # Set up the mask 
  mask_shape = tf.stack([batch_size, bucket_size, 1])
  mask = nn.binary_mask(mask_shape, embed_keep_prob)
  
  # Replace with random vector
  var = tf.nn.moments(layer, [0,1,2])[1]+1e-12
  maxval = tf.sqrt(3*var)
  rand = tf.random_uniform(mask_shape, minval=-maxval, maxval=maxval)
  layer = mask*layer + (1-mask)*rand
  
  return layer

#===============================================================
def token_embedding_lookup(n_entries, embed_size, ids, nonzero_init=False, reuse=False):
  """"""
  
  shape = [n_entries, embed_size]
  initializer = tf.random_normal_initializer if nonzero_init else tf.zeros_initializer
  with tf.device('/cpu:0'):
    params = tf.get_variable('Embeddings', shape=shape, initializer=initializer)
  layers = tf.nn.embedding_lookup(params, ids)
  return layers

#===============================================================
def pretrained_embedding_lookup(params, linear_size, ids, name='', reuse=True):
  """"""
  
  layer = tf.nn.embedding_lookup(params, ids)
  batch_size, bucket_size, input_size = nn.get_sizes(layer)
  shape = [input_size, linear_size]
  weights = tf.get_variable(name+'Transformation', shape=shape, initializer=tf.orthogonal_initializer)
  layer = tf.reshape(layer, [-1, input_size])
  layer = tf.matmul(layer, weights)
  layer = tf.reshape(layer, tf.stack([batch_size, bucket_size, linear_size]))
  return layer

#===============================================================
def concat(layers, embed_keep_prob=1., drop_func=dropout, reuse=True):
  """"""
  
  layer = tf.concat(layers, 2)
  
  if embed_keep_prob < 1:
    layer = drop_func(layer, embed_keep_prob)
  
  ## Upscale everything
  #if embed_keep_prob < 1 and drop_func == dropout:
  #  output_size = layer.get_shape().as_list()[-1]
  #  n_nonzeros = tf.count_nonzero(layer, axis=2, keep_dims=True)
  #  scale_factor = output_size / (n_nonzeros+1e-12)
  #  layer *= scale_factor
  
  return layer

#===============================================================
def reduce_max(layers, embed_keep_prob=1., drop_func=dropout, reuse=True):
  """"""
  
  layer = tf.reduce_max(tf.stack(layers), 0)
  
  if embed_keep_prob < 1:
    layer = drop_func(layer, embed_keep_prob)
  
  return layer

#===============================================================
def reduce_sum(layers, embed_keep_prob=1., drop_func=dropout, reuse=True):
  """"""
  
  layer = tf.add_n(layers)
  
  if embed_keep_prob < 1:
    layer = drop_func(layer, embed_keep_prob)
  
  return layer

#===============================================================
def gated(layers, embed_keep_prob=1., drop_func=dropout, reuse=True):
  """"""
  
  _, layer = linear_attention(tf.stack(layers, axis=-2)) # axis=-1?
  
  if embed_keep_prob < 1:
    layer = drop_func(layer, embed_keep_prob)
  
  return layer
