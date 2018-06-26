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
import six

import numpy as np
import tensorflow as tf

from .base_vocabs import BaseVocab 
from . import conllu_vocabs as cv

from parser.neural import nn, nonlin, classifiers

#***************************************************************
class IndexVocab(BaseVocab):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    super(IndexVocab, self).__init__(*args, **kwargs)
    
    self.PAD_STR = '_'
    self.PAD_IDX = -1
    self.ROOT_STR = '0'
    self.ROOT_IDX = 0
    return
  
  #=============================================================
  def add(self, token):
    """"""
    
    return self.index(token)
  
  #=============================================================
  def token(self, index):
    """"""
    
    if index > -1:
      return str(index)
    else:
      return '_'
  
  #=============================================================
  def index(self, token):
    """"""
    
    if token != '_':
      return int(token)
    else:
      return -1
  
  #=============================================================
  def get_root(self):
    """"""
    
    return self.ROOT_STR
  
  #=============================================================
  def get_bilinear_classifier(self, layer, token_weights, variable_scope=None, reuse=False):
    """"""
    
    recur_layer = layer1 = layer2 = layer
    lin_layer1 = lin_layer2 = layer
    hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
    hidden_func = self.hidden_func
    hidden_size = self.hidden_size
    add_linear = self.add_linear
    linearize = self.linearize
    distance = self.distance
    n_splits = 2*(1+linearize+distance)
    with tf.variable_scope(variable_scope or self.field):
      for i in six.moves.range(0, self.n_layers-1):
        with tf.variable_scope('FC-%d' % i):
          layer = classifiers.hidden(layer, n_splits*hidden_size,
                                    hidden_func=hidden_func,
                                    hidden_keep_prob=hidden_keep_prob)
      with tf.variable_scope('FC-top'):
        layers = classifiers.hiddens(layer, n_splits*[hidden_size],
                                    hidden_func=hidden_func,
                                    hidden_keep_prob=hidden_keep_prob)
      layer1, layer2 = layers.pop(0), layers.pop(0)
      if linearize:
        lin_layer1, lin_layer2 = layers.pop(0), layers.pop(0)
      if distance:
        dist_layer1, dist_layer2 = layers.pop(0), layers.pop(0)
      
      with tf.variable_scope('Attention'):
        if self.diagonal:
          logits, _ = classifiers.diagonal_bilinear_attention(
            layer1, layer2, 
            hidden_keep_prob=hidden_keep_prob,
            add_linear=add_linear)
          if linearize:
            with tf.variable_scope('Linearization'):
              lin_logits = classifiers.diagonal_bilinear_discriminator(
                lin_layer1, lin_layer2,
                hidden_keep_prob=hidden_keep_prob,
                add_linear=add_linear)
          if distance:
            with tf.variable_scope('Distance'):
              dist_lamda = 1+tf.nn.softplus(classifiers.diagonal_bilinear_discriminator(
                dist_layer1, dist_layer2,
                hidden_keep_prob=hidden_keep_prob,
                add_linear=add_linear))
        else:
          logits, _ = classifiers.bilinear_attention(
            layer1, layer2,
            hidden_keep_prob=hidden_keep_prob,
            add_linear=add_linear)
          if linearize:
            with tf.variable_scope('Linearization'):
              lin_logits = classifiers.bilinear_discriminator(
                lin_layer1, lin_layer2,
                hidden_keep_prob=hidden_keep_prob,
                add_linear=add_linear)
          if distance:
            with tf.variable_scope('Distance'):
              dist_lamda = 1+tf.nn.softplus(classifiers.bilinear_discriminator(
                dist_layer1, dist_layer2,
                hidden_keep_prob=hidden_keep_prob,
                add_linear=add_linear))
        
        #-----------------------------------------------------------
        # Process the targets
        targets = self.placeholder
        shape = tf.shape(layer1)
        batch_size, bucket_size = shape[0], shape[1]
        # (1 x m)
        ids = tf.expand_dims(tf.range(bucket_size), 0)
        # (1 x m) -> (1 x 1 x m)
        head_ids = tf.expand_dims(ids, -2)
        # (1 x m) -> (1 x m x 1)
        dep_ids = tf.expand_dims(ids, -1)
        if linearize:
          # Wherever the head is to the left
          # (n x m), (1 x m) -> (n x m)
          lin_targets = tf.to_float(tf.less(targets, ids))
          # cross-entropy of the linearization of each i,j pair
          # (1 x 1 x m), (1 x m x 1) -> (n x m x m)
          lin_ids = tf.tile(tf.less(head_ids, dep_ids), [batch_size, 1, 1])
          # (n x 1 x m), (n x m x 1) -> (n x m x m)
          lin_xent = -tf.nn.softplus(tf.where(lin_ids, -lin_logits, lin_logits))
          # add the cross-entropy to the logits
          # (n x m x m), (n x m x m) -> (n x m x m)
          logits += tf.stop_gradient(lin_xent)
        if distance:
          # (n x m) - (1 x m) -> (n x m)
          dist_targets = tf.abs(targets - ids)
          # KL-divergence of the distance of each i,j pair
          # (1 x 1 x m) - (1 x m x 1) -> (n x m x m)
          dist_ids = tf.to_float(tf.tile(tf.abs(head_ids - dep_ids), [batch_size, 1, 1]))+1e-12
          # (n x m x m), (n x m x m) -> (n x m x m)
          #dist_kld = (dist_ids * tf.log(dist_lamda / dist_ids) + dist_ids - dist_lamda)
          dist_kld = -2*tf.log(tf.abs(dist_ids - dist_lamda) + 1)
          # add the KL-divergence to the logits
          # (n x m x m), (n x m x m) -> (n x m x m)
          logits += tf.stop_gradient(dist_kld)
        
        #-----------------------------------------------------------
        # Compute probabilities/cross entropy
        # (n x m) + (m) -> (n x m)
        non_pads = tf.to_float(token_weights) + tf.to_float(tf.logical_not(tf.cast(tf.range(bucket_size), dtype=tf.bool)))
        # (n x m x m) o (n x 1 x m) -> (n x m x m)
        probabilities = tf.nn.softmax(logits) * tf.expand_dims(non_pads, -2)
        # (n x m), (n x m x m), (n x m) -> ()
        loss = tf.losses.sparse_softmax_cross_entropy(
          targets,
          logits,
          weights=token_weights)
        # (n x m) -> (n x m x m x 1)
        one_hot_targets = tf.expand_dims(tf.one_hot(targets, bucket_size), -1)
        # (n x m) -> ()
        n_tokens = tf.to_float(tf.reduce_sum(token_weights))
        if linearize:
          # (n x m x m) -> (n x m x 1 x m)
          lin_xent_reshaped = tf.expand_dims(lin_xent, -2)
          # (n x m x 1 x m) * (n x m x m x 1) -> (n x m x 1 x 1)
          lin_target_xent = tf.matmul(lin_xent_reshaped, one_hot_targets)
          # (n x m x 1 x 1) -> (n x m)
          lin_target_xent = tf.squeeze(lin_target_xent, [-1, -2])
          # (n x m), (n x m), (n x m) -> ()
          loss -= tf.reduce_sum(lin_target_xent*tf.to_float(token_weights)) / (n_tokens + 1e-12)
        if distance:
          # (n x m x m) -> (n x m x 1 x m)
          dist_kld_reshaped = tf.expand_dims(dist_kld, -2)
          # (n x m x 1 x m) * (n x m x m x 1) -> (n x m x 1 x 1)
          dist_target_kld = tf.matmul(dist_kld_reshaped, one_hot_targets)
          # (n x m x 1 x 1) -> (n x m)
          dist_target_kld = tf.squeeze(dist_target_kld, [-1, -2])
          # (n x m), (n x m), (n x m) -> ()
          loss -= tf.reduce_sum(dist_target_kld*tf.to_float(token_weights)) / (n_tokens + 1e-12)
        
        #-----------------------------------------------------------
        # Compute predictions/accuracy
        # (n x m x m) -> (n x m)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        # (n x m) (*) (n x m) -> (n x m)
        correct_tokens = nn.equal(targets, predictions) * token_weights
        # (n x m) -> (n)
        tokens_per_sequence = tf.reduce_sum(token_weights, axis=-1)
        # (n x m) -> (n)
        correct_tokens_per_sequence = tf.reduce_sum(correct_tokens, axis=-1)
        # (n), (n) -> (n)
        correct_sequences = nn.equal(tokens_per_sequence, correct_tokens_per_sequence)
    
    #-----------------------------------------------------------
    # Populate the output dictionary
    outputs = {}
    outputs['recur_layer'] = recur_layer
    outputs['unlabeled_targets'] = self.placeholder
    outputs['probabilities'] = probabilities
    outputs['unlabeled_loss'] = loss
    outputs['loss'] = loss
    
    outputs['unlabeled_predictions'] = predictions
    outputs['predictions'] = predictions
    outputs['correct_unlabeled_tokens'] = correct_tokens
    outputs['n_correct_unlabeled_tokens'] = tf.reduce_sum(correct_tokens)
    outputs['n_correct_unlabeled_sequences'] = tf.reduce_sum(correct_sequences)
    outputs['n_correct_tokens'] = tf.reduce_sum(correct_tokens)
    outputs['n_correct_sequences'] = tf.reduce_sum(correct_sequences)
    return outputs
  
  #=============================================================
  def __getitem__(self, key):
    if isinstance(key, six.string_types):
      if key == '_':
        return -1
      else:
        return int(key)
    elif isinstance(key, six.integer_types + (np.int32, np.int64)):
      if key > -1:
        return str(key)
      else:
        return '_'
    elif hasattr(key, '__iter__'):
      return [self[k] for k in key]
    else:
      raise ValueError('key to IndexVocab.__getitem__ must be (iterable of) string or integer')
    return
  #=============================================================
  @property
  def distance(self):
    return self._config.getboolean(self, 'distance')
  @property
  def linearize(self):
    return self._config.getboolean(self, 'linearize')
  @property
  def decomposition_level(self):
    return self._config.getint(self, 'decomposition_level')
  @property
  def diagonal(self):
    return self._config.getboolean(self, 'diagonal')
  @property
  def add_linear(self):
    return self._config.getboolean(self, 'add_linear')
  @property
  def n_layers(self):
    return self._config.getint(self, 'n_layers')
  @property
  def hidden_size(self):
    return self._config.getint(self, 'hidden_size')
  @property
  def hidden_keep_prob(self):
    return self._config.getfloat(self, 'hidden_keep_prob')
  @property
  def hidden_func(self):
    hidden_func = self._config.getstr(self, 'hidden_func')
    if hasattr(nonlin, hidden_func):
      return getattr(nonlin, hidden_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, hidden_func))
  
#***************************************************************
class GraphIndexVocab(IndexVocab):
  """"""
  
  _depth = -1
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    kwargs['placeholder_shape'] = [None, None, None]
    super(GraphIndexVocab, self).__init__(*args, **kwargs)
    return
  
  #=============================================================
  def get_bilinear_discriminator(self, layer, token_weights, variable_scope=None, reuse=False):
    """"""
    
    layer1 = layer2 = layer
    hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
    add_linear = self.add_linear
    with tf.variable_scope(variable_scope or self.field):
      for i in six.moves.range(0, self.n_layers):
        with tf.variable_scope('FC1-%d' % i):
          layer1 = classifiers.hidden(layer1, self.hidden_size,
                                      hidden_func=self.hidden_func,
                                      hidden_keep_prob=hidden_keep_prob)
        with tf.variable_scope('FC2-%d' % i):
          layer2 = classifiers.hidden(layer2, self.hidden_size,
                                      hidden_func=self.hidden_func,
                                      hidden_keep_prob=hidden_keep_prob)
      with tf.variable_scope('Discriminator'):
        if self.diagonal:
          logits = classifiers.diagonal_full_bilinear_discriminator(
            layer1, layer2,
            hidden_keep_prob=hidden_keep_prob,
            add_linear=add_linear)
        else:
          logits = classifiers.full_bilinear_discriminator(
            layer1, layer2,
            hidden_keep_prob=hidden_keep_prob,
            add_linear=add_linear)
        
        #-----------------------------------------------------------
        # Process the targets
        # (n x m x m) -> (n x m x m)
        unlabeled_targets = self.placeholder
        
        #-----------------------------------------------------------
        # Compute probabilities/cross entropy
        # (n x m x m) -> (n x m x m)
        probabilities = tf.nn.sigmoid(logits) * tf.to_float(token_weights)
        # (n x m x m), (n x m x m), (n x m x m) -> ()
        loss = tf.losses.sigmoid_cross_entropy(unlabeled_targets, logits, weights=token_weights)
        
        #-----------------------------------------------------------
        # Compute predictions/accuracy
        # (n x m x m) -> (n x m x m)
        predictions = nn.greater(logits, 0, dtype=tf.int32) * token_weights
        # (n x m x m) (*) (n x m x m) -> (n x m x m)
        true_positives = predictions * unlabeled_targets
        # (n x m x m) -> ()
        n_predictions = tf.reduce_sum(predictions)
        n_targets = tf.reduce_sum(unlabeled_targets)
        n_true_positives = tf.reduce_sum(true_positives)
        # () - () -> ()
        n_false_positives = n_predictions - n_true_positives
        n_false_negatives = n_targets - n_true_positives
        # (n x m x m) -> (n)
        n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1,2])
        n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1,2])
        # (n) x 2 -> ()
        n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))
    
    #-----------------------------------------------------------
    # Populate the output dictionary
    outputs = {}
    outputs['unlabeled_targets'] = unlabeled_targets
    outputs['probabilities'] = probabilities
    outputs['unlabeled_loss'] = loss
    outputs['loss'] = loss
    
    outputs['unlabeled_predictions'] = predictions
    outputs['n_unlabeled_true_positives'] = n_true_positives
    outputs['n_unlabeled_false_positives'] = n_false_positives
    outputs['n_unlabeled_false_negatives'] = n_false_negatives
    outputs['n_correct_unlabeled_sequences'] = n_correct_sequences
    outputs['predictions'] = predictions
    outputs['n_true_positives'] = n_true_positives
    outputs['n_false_positives'] = n_false_positives
    outputs['n_false_negatives'] = n_false_negatives
    outputs['n_correct_sequences'] = n_correct_sequences
    return outputs
  
  #=============================================================
  # token should be: 1:rel|2:acl|5:dep or 1|2|5
  def index(self, token):
    """"""
    
    nodes = []
    if token != '_':
      token = token.split('|')
      for edge in token:
        head = edge.split(':')[0]
        nodes.append(int(head))
    return nodes
  
  #=============================================================
  # index should be [1, 2, 5]
  def token(self, index):
    """"""
    
    return [str(head) for head in index]
  
  #=============================================================
  def get_root(self):
    """"""
    
    return '_'
  
  #=============================================================
  def __getitem__(self, key):
    if isinstance(key, six.string_types):
      nodes = []
      if key != '_':
        token = key.split('|')
        for edge in token:
          head = edge.split(':')[0]
          nodes.append(int(head))
      return nodes
    elif hasattr(key, '__iter__'):
      if len(key) > 0:
        if isinstance(key[0], six.integer_types + (np.int32, np.int64)):
          return '|'.join([str(head) for head in key])
        else:
          return [self[k] for k in key]
      else:
        return '_'
    else:
      raise ValueError('Key to GraphIndexVocab.__getitem__ must be (iterable of) strings or iterable of integers')
  
#***************************************************************
class IDIndexVocab(IndexVocab, cv.IDVocab):
  pass
class DepheadIndexVocab(IndexVocab, cv.DepheadVocab):
  pass
class SemheadGraphIndexVocab(GraphIndexVocab, cv.SemheadVocab):
  pass
