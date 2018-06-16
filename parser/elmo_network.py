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

import re
import os
import pickle as pkl
import curses
import codecs

import numpy as np
import tensorflow as tf

from parser.base_network import BaseNetwork
from parser.neural import nn, nonlin, embeddings, recurrent, classifiers

#***************************************************************
class ElmoNetwork(BaseNetwork):
  """"""
  
  _prefix_root = False
  _postfix_root = False
  _evals = []
  
  #=============================================================
  def build_graph(self, input_network_outputs={}, reuse=True):
    """"""
    
    outputs = {}
    with tf.variable_scope('Embeddings'):
      input_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs]
      for input_network, output in input_network_outputs:
        with tf.variable_scope(input_network.classname):
          input_tensors.append(input_network.get_input_tensor(output, reuse=reuse))
      layer = tf.concat(input_tensors, 2)
    n_nonzero = tf.to_float(tf.count_nonzero(layer, axis=-1, keep_dims=True))
    batch_size, bucket_size, input_size = nn.get_sizes(layer)
    layer *= input_size / (n_nonzero + tf.constant(1e-12))
    
    token_weights = nn.greater(self.id_vocab.placeholder, 0)
    tokens_per_sequence = tf.reduce_sum(token_weights, axis=1)
    n_tokens = tf.reduce_sum(tokens_per_sequence)
    n_sequences = tf.count_nonzero(tokens_per_sequence)
    seq_lengths = tokens_per_sequence + self.prefix_root+self.postfix_root
    tokens = {'n_tokens': n_tokens,
              'tokens_per_sequence': tokens_per_sequence,
              'token_weights': token_weights,
              'n_sequences': n_sequences}
    
    conv_keep_prob = 1. if reuse else self.conv_keep_prob
    recur_keep_prob = 1. if reuse else self.recur_keep_prob
    recur_include_prob = 1. if reuse else self.recur_include_prob
    
    rev_layer = tf.reverse_sequence(layer, seq_lengths, seq_axis=2)
    for i in six.moves.range(self.n_layers):
      conv_width = self.first_layer_conv_width if not i else self.conv_width
      with tf.variable_scope('RNN_FW-{}'.format(i)):
        layer, _ = recurrent.directed_RNN(layer, self.recur_size, seq_lengths,
                                          bidirectional=False,
                                          recur_cell=self.recur_cell,
                                          conv_width=conv_width,
                                          recur_func=self.recur_func,
                                          conv_keep_prob=conv_keep_prob,
                                          recur_include_prob=recur_include_prob,
                                          recur_keep_prob=recur_keep_prob,
                                          cifg=self.cifg,
                                          highway=self.highway,
                                          highway_func=self.highway_func)
      if self.bidirectional:
        with tf.variable_scope('RNN_BW-{}'.format(i)):
          rev_layer, _ = recurrent.directed_RNN(rev_layer, self.recur_size, seq_lengths,
                                                bidirectional=False,
                                                recur_cell=self.recur_cell,
                                                conv_width=conv_width,
                                                recur_func=self.recur_func,
                                                conv_keep_prob=conv_keep_prob,
                                                recur_keep_prob=recur_keep_prob,
                                                recur_include_prob=recur_include_prob,
                                                cifg=self.cifg,
                                                highway=self.highway,
                                                highway_func=self.highway_func)
    ones = tf.ones([batch_size, 1, 1])
    with tf.variable_scope('RNN_FW-{}/RNN/Loop'.format(i), reuse=True):
      fw_initial_state = tf.get_variable('Initial_state')
      n_splits = fw_initial_state.get_shape().as_list()[-1] / self.recur_size
      fw_initial_state = tf.split(fw_initial_state, int(n_splits), -1)[0]
      start_token = ones * fw_initial_state
      layer = tf.reverse_sequence(layer, seq_lengths, seq_axis=2)
      layer = layer[:,1:]
      layer = tf.reverse_sequence(layer, seq_lengths-1, seq_axis=2)
      layer = tf.concat([start_token, layer], axis=1)
    if self.bidirectional:
      with tf.variable_scope('RNN_BW-{}/RNN/Loop'.format(i), reuse=True):
        bw_initial_state = tf.get_variable('Initial_state')
        n_splits = bw_initial_state.get_shape().as_list()[-1] / self.recur_size
        bw_initial_state = tf.split(bw_initial_state, int(n_splits), -1)[0]
        stop_token = ones * bw_initial_state
        rev_layer = tf.concat([stop_token, layer], axis=1)
        rev_layer = tf.reverse_sequence(rev_layer, seq_lengths+1, seq_axis=2)[:,1:]
      if self.bilin:
        layer = tf.concat([layer*rev_layer, layer, rev_layer], axis=2)
      else:
        layer = tf.concat([layer, rev_layer], axis=2)
    #input_vocabs = {vocab.field: vocab for vocab in self.input_vocabs}
    output_vocabs = {vocab.field: vocab for vocab in self.output_vocabs}
    outputs = {}
    with tf.variable_scope('Classifiers'):
      if 'form' in output_vocabs:
        vocab = output_vocabs['form']
        outputs[vocab.field] = vocab.get_linear_classifier(
          layer,
          token_weights=token_weights,
          reuse=reuse)
        self._evals.add('form')
      if 'upos' in output_vocabs:
        vocab = output_vocabs['upos']
        outputs[vocab.field] = vocab.get_linear_classifier(
          layer,
          token_weights=token_weights,
          reuse=reuse)
        self._evals.add('upos')
      if 'xpos' in output_vocabs:
        vocab = output_vocabs['xpos']
        outputs[vocab.field] = vocab.get_linear_classifier(
          layer,
          token_weights=token_weights,
          reuse=reuse)
        self._evals.add('xpos')
    return outputs, tokens
  
