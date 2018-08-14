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
class GraphParserNetwork(BaseNetwork):
  """"""
  
  #=============================================================
  def build_graph(self, input_network_outputs={}, reuse=True):
    """"""
    
    with tf.variable_scope('Embeddings'):
      if self.sum_pos: # TODO this should be done with a `POSMultivocab`
        pos_vocabs = list(filter(lambda x: 'POS' in x.classname, self.input_vocabs))
        pos_tensors = [input_vocab.get_input_tensor(embed_keep_prob=1, reuse=reuse) for input_vocab in pos_vocabs]
        non_pos_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs if 'POS' not in input_vocab.classname]
        #pos_tensors = [tf.Print(pos_tensor, [pos_tensor]) for pos_tensor in pos_tensors]
        #non_pos_tensors = [tf.Print(non_pos_tensor, [non_pos_tensor]) for non_pos_tensor in non_pos_tensors]
        if pos_tensors:
          pos_tensors = tf.add_n(pos_tensors)
          if not reuse:
            pos_tensors = [pos_vocabs[0].drop_func(pos_tensors, pos_vocabs[0].embed_keep_prob)]
          else:
            pos_tensors = [pos_tensors]
        input_tensors = non_pos_tensors + pos_tensors
      else:
        input_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs]
      for input_network, output in input_network_outputs:
        with tf.variable_scope(input_network.classname):
          input_tensors.append(input_network.get_input_tensor(output, reuse=reuse))
      layer = tf.concat(input_tensors, 2)

    n_nonzero = tf.to_float(tf.count_nonzero(layer, axis=-1, keepdims=True))
    batch_size, bucket_size, input_size = nn.get_sizes(layer)
    layer *= input_size / (n_nonzero + tf.constant(1e-12))
    
    token_weights = nn.greater(self.id_vocab.placeholder, 0)
    tokens_per_sequence = tf.reduce_sum(token_weights, axis=1)
    n_tokens = tf.reduce_sum(tokens_per_sequence)
    n_sequences = tf.count_nonzero(tokens_per_sequence)
    seq_lengths = tokens_per_sequence+1

    root_weights = token_weights + (1-nn.greater(tf.range(bucket_size), 0))
    token_weights3D = tf.expand_dims(token_weights, axis=-1) * tf.expand_dims(root_weights, axis=-2)
    tokens = {'n_tokens': n_tokens,
              'tokens_per_sequence': tokens_per_sequence,
              'token_weights': token_weights,
              'token_weights3D': token_weights,
              'n_sequences': n_sequences}
    
    conv_keep_prob = 1. if reuse else self.conv_keep_prob
    recur_keep_prob = 1. if reuse else self.recur_keep_prob
    recur_include_prob = 1. if reuse else self.recur_include_prob
    
    for i in six.moves.range(self.n_layers):
      conv_width = self.first_layer_conv_width if not i else self.conv_width
      with tf.variable_scope('RNN-{}'.format(i)):
        layer, _ = recurrent.directed_RNN(layer, self.recur_size, seq_lengths,
                                          bidirectional=self.bidirectional,
                                          recur_cell=self.recur_cell,
                                          conv_width=conv_width,
                                          recur_func=self.recur_func,
                                          conv_keep_prob=conv_keep_prob,
                                          recur_include_prob=recur_include_prob,
                                          recur_keep_prob=recur_keep_prob,
                                          cifg=self.cifg,
                                          highway=self.highway,
                                          highway_func=self.highway_func,
                                          bilin=self.bilin)
  
    output_fields = {vocab.field: vocab for vocab in self.output_vocabs}
    outputs = {}
    with tf.variable_scope('Classifiers'):
      if 'semrel' in output_fields:
        vocab = output_fields['semrel']
        head_vocab = output_fields['semhead']
        if vocab.factorized:
          with tf.variable_scope('Unlabeled'):
            unlabeled_outputs = head_vocab.get_bilinear_discriminator(
              layer,
              token_weights=token_weights3D,
              reuse=reuse)
          with tf.variable_scope('Labeled'):
            labeled_outputs = vocab.get_bilinear_classifier(
              layer, unlabeled_outputs,
              token_weights=token_weights3D,
              reuse=reuse)
        else:
          labeled_outputs = vocab.get_unfactored_bilinear_classifier(layer, head_vocab.placeholder,
            token_weights=token_weights3D,
            reuse=reuse)
        outputs['semgraph'] = labeled_outputs
        self._evals.add('semgraph')
      elif 'semhead' in output_fields:
        vocab = output_fields['semhead']
        outputs[vocab.classname] = vocab.get_bilinear_classifier(
          layer,
          token_weights=token_weights3D,
          reuse=reuse)
        self._evals.add('semhead')
    
    return outputs, tokens
  
  #=============================================================
  @property
  def sum_pos(self):
    return self._config.getboolean(self, 'sum_pos')
