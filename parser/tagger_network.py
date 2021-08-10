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
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from debug.timer import Timer

from parser.base_network import BaseNetwork
from parser.neural import nn, nonlin, embeddings, recurrent, classifiers

#***************************************************************
class TaggerNetwork(BaseNetwork):
  """"""
  
  #=============================================================
  def build_graph(self, input_network_outputs={}, reuse=True):
    """"""
    
    with tf.variable_scope('Embeddings'):
      input_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs]
      for input_network, output in input_network_outputs:
        with tf.variable_scope(input_network.classname):
          input_tensors.append(input_network.get_input_tensor(output, reuse=reuse))
      layer = tf.concat(input_tensors, 2)
    batch_size, bucket_size, input_size = nn.get_sizes(layer)
    n_nonzero = tf.to_float(tf.count_nonzero(layer, axis=-1, keep_dims=True))
    layer *= input_size / (n_nonzero + tf.constant(1e-12))
    
    token_weights = nn.greater(self.id_vocab.placeholder, 0)
    tokens_per_sequence = tf.reduce_sum(token_weights, axis=1)
    n_tokens = tf.reduce_sum(tokens_per_sequence)
    n_sequences = tf.count_nonzero(tokens_per_sequence)
    seq_lengths = tokens_per_sequence + 1
    tokens = {'n_tokens': n_tokens,
              'tokens_per_sequence': tokens_per_sequence,
              'token_weights': token_weights,
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
                                          recur_keep_prob=recur_keep_prob,
                                          recur_include_prob=recur_include_prob,
                                          cifg=self.cifg,
                                          highway=self.highway,
                                          highway_func=self.highway_func,
                                          bilin=self.bilin)
    
    output_vocabs = {vocab.field: vocab for vocab in self.output_vocabs}
    outputs = {}
    with tf.variable_scope('Classifiers'):
      last_output = None
      if 'lemma' in output_vocabs:
        vocab = output_vocabs['lemma']
        outputs[vocab.field] = vocab.get_linear_classifier(
          layer, token_weights,
          last_output if self.share_layer else None,
          reuse=reuse)
        self._evals.add('lemma')
        if last_output is None:
          last_output = outputs[vocab.field]
      if 'upos' in output_vocabs:
        vocab = output_vocabs['upos']
        outputs[vocab.field] = vocab.get_linear_classifier(
          layer, token_weights,
          last_output if self.share_layer else None, 
          reuse=reuse)
        self._evals.add('upos')
        if last_output is None:
          last_output = outputs[vocab.field]
        if reuse:
          upos_idxs = outputs[vocab.field]['predictions'] 
        else:
          upos_idxs = outputs[vocab.field]['targets']
        upos_embed = vocab.get_input_tensor(inputs=upos_idxs, embed_keep_prob=1, reuse=reuse)
        if 'xpos' in output_vocabs and not self.share_layer:
          vocab = output_vocabs['xpos']
          outputs[vocab.field] = vocab.get_bilinear_classifier_with_embeddings(
            layer, upos_embed, token_weights,
            reuse=reuse)
          self._evals.add('xpos')
        if 'ufeats' in output_vocabs and not self.share_layer:
          vocab = output_vocabs['ufeats']
          outputs[vocab.field] = vocab.get_bilinear_classifier_with_embeddings(
            layer, upos_embed, token_weights,
            reuse=reuse)
          self._evals.add('ufeats')
        #if 'ufeats' in output_vocabs and not self.share_layer:
        #  vocab = output_vocabs['ufeats']
        #  outputs[vocab.field] = vocab.get_bilinear_classifier_with_embeddings(
        #    layer, upos_embed, token_weights,
        #    reuse=reuse)
        #  self._evals.add('ufeats')
      if 'xpos' in output_vocabs and ('upos' not in output_vocabs or self.share_layer):
        vocab = output_vocabs['xpos']
        outputs[vocab.field] = vocab.get_linear_classifier(
          layer, token_weights,
          last_output if self.share_layer else None, 
          reuse=reuse)
        self._evals.add('xpos')
        if last_output is None:
          last_output = outputs[vocab.field]
      if 'ufeats' in output_vocabs and ('upos' not in output_vocabs or self.share_layer):
      #if 'ufeats' in output_vocabs and ('upos' not in output_vocabs or self.share_layer):
        vocab = output_vocabs['ufeats']
        outputs[vocab.field] = vocab.get_linear_classifier(
          layer, token_weights,
          last_output if self.share_layer else None, 
          reuse=reuse)
        self._evals.add('ufeats')
        if last_output is None:
          last_output = outputs[vocab.field]
      if 'deprel' in output_vocabs:
        vocab = output_vocabs['deprel']
        outputs[vocab.field] = vocab.get_linear_classifier(
          layer, token_weights,
          last_output if self.share_layer else None, 
          reuse=reuse)
        self._evals.add('deprel')
        if last_output is None:
          last_output = outputs[vocab.field]
    return outputs, tokens
