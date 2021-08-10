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
import codecs
from collections import Counter

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from parser.structs.buckets import ListMultibucket
from .base_vocabs import CountVocab
from . import conllu_vocabs as cv

from parser.neural import nn, nonlin, embeddings, recurrent, classifiers

#***************************************************************
class SubtokenVocab(CountVocab):
  """"""

  _save_str = 'subtokens'

  #=============================================================
  def __init__(self, config=None):
    """"""

    super(SubtokenVocab, self).__init__(config=config)
    self._multibucket = ListMultibucket(self, max_buckets=self.max_buckets, config=config)
    self._tok2idx = {}
    self._idx2tok = {}
    return

  #=============================================================
  def get_input_tensor(self, embed_keep_prob=None, nonzero_init=False, variable_scope=None, reuse=True):
    """"""

    embed_keep_prob = embed_keep_prob or self.embed_keep_prob
    conv_keep_prob = 1. if reuse else self.conv_keep_prob
    recur_keep_prob = 1. if reuse else self.recur_keep_prob
    output_keep_prob = 1. if reuse else self.output_keep_prob

    layers = []
    with tf.variable_scope(variable_scope or self.classname) as scope:
      for i, placeholder in enumerate(self._multibucket.get_placeholders()):
        if i:
          scope.reuse_variables()
        #with tf.device('/gpu:0'):
        #with tf.device('/gpu:{}'.format(i)):
        with tf.variable_scope('Embeddings'):
          layer = embeddings.token_embedding_lookup(len(self), self.embed_size,
                                                     placeholder,
                                                     nonzero_init=True,
                                                     reuse=reuse)

        seq_lengths = tf.count_nonzero(placeholder, axis=1, dtype=tf.int32)
        for j in six.moves.range(self.n_layers):
          conv_width = self.first_layer_conv_width if not j else self.conv_width
          with tf.variable_scope('RNN-{}'.format(j)):
            layer, final_states = recurrent.directed_RNN(
              layer, self.recur_size, seq_lengths,
              bidirectional=self.bidirectional,
              recur_cell=self.recur_cell,
              conv_width=conv_width,
              recur_func=self.recur_func,
              conv_keep_prob=conv_keep_prob,
              recur_keep_prob=recur_keep_prob,
              cifg=self.cifg,
              highway=self.highway,
              highway_func=self.highway_func,
              bilin=self.bilin)


        if not self.squeeze_type.startswith('gated'):
          if self.squeeze_type == 'linear_attention':
            with tf.variable_scope('Attention'):
              _, layer = classifiers.linear_attention(layer, hidden_keep_prob=output_keep_prob)
          elif self.squeeze_type == 'final_hidden':
            layer, _ = tf.split(final_states, 2, axis=-1)
          elif self.squeeze_type == 'final_cell':
            _, layer = tf.split(final_states, 2, axis=-1)
          elif self.squeeze_type == 'final_state':
            layer = final_states
          elif self.squeeze_type == 'reduce_max':
            layer = tf.reduce_max(layer, axis=-2)
          with tf.variable_scope('Linear'):
            layer = classifiers.hidden(layer, self.output_size,
                                       hidden_func=self.output_func,
                                       hidden_keep_prob=output_keep_prob)
        else:
          with tf.variable_scope('Attention'):
            attn, layer = classifiers.deep_linear_attention(layer, self.output_size,
                                       hidden_func=nonlin.identity,
                                       hidden_keep_prob=output_keep_prob)
          if self.squeeze_type == 'gated_reduce_max':
            layer = tf.nn.relu(tf.reduce_max(layer, axis=-2)) + .1*tf.reduce_sum(layer, axis=-2)/(tf.count_nonzero(layer, axis=-2, dtype=tf.float32)+1e-12)
          elif self.squeeze_type == 'gated_reduce_sum':
            layer = self.output_func(tf.reduce_sum(layer, axis=-2))
        #layer = tf.tf.Print(layer, [tf.shape(layer)])
        layers.append(layer)
      # Concatenate all the buckets' embeddings
      layer = tf.concat(layers, 0)
      # Put them in the right order, creating the embedding matrix
      layer = tf.nn.embedding_lookup(layer, self._multibucket.placeholder)
      #layer = tf.nn.embedding_lookup(layers, self._multibucket.placeholder, partition_strategy='div')
      #layer = tf.Print(layer, [tf.shape(layer)])
      # Get the embeddings from the embedding matrix
      layer = tf.nn.embedding_lookup(layer, self.placeholder)

      if embed_keep_prob < 1:
        layer = self.drop_func(layer, embed_keep_prob)
    return layer

  #=============================================================
  def count(self, train_conllus):
    """"""

    tokens = set()
    for train_conllu in train_conllus:
      with codecs.open(train_conllu, encoding='utf-8', errors='ignore') as f:
        for line in f:
          line = line.strip()
          if line and not line.startswith('#'):
            line = line.split('\t')
            token = line[self.conllu_idx] # conllu_idx is provided by the CoNLLUVocab
            if token not in tokens:
              tokens.add(token)
              self._count(token)
    self.index_by_counts()
    return True

  def _count(self, token):
    if not self.cased:
      token = token.lower()
    self.counts.update(token)
    return

  #=============================================================
  def load(self):
    """"""

    if super(SubtokenVocab, self).load():
      self._loaded = True
      return True
    else:
      if os.path.exists(self.token_vocab_savename):
        token_vocab_filename = self.token_vocab_savename
      elif self.token_vocab_loadname and os.path.exists(self.token_vocab_loadname):
        token_vocab_filename = self.token_vocab_loadname
      else:
        self._loaded = False
        return False

      with codecs.open(token_vocab_filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
          line = line.rstrip()
          if line:
            match = re.match('(.*)\s([0-9]*)', line)
            token = match.group(1)
            count = int(match.group(2))
            self._count(token)
            self._count(token.upper())
      self.index_by_counts(dump=True)
      self._loaded = True
      return True

  #=============================================================
  def add(self, token):
    """"""

    characters = list(token)
    character_indices = [self._str2idx.get(character, self.UNK_IDX) for character in characters[:50]]
    token_index = self._multibucket.add(character_indices, characters)
    self._tok2idx[token] = token_index
    self._idx2tok[token_index] = token
    return token_index

  #=============================================================
  def token(self, index):
    """"""

    return self._idx2tok[index]

  #=============================================================
  def index(self, token):
    """"""

    return self._tok2idx[token]

  #=============================================================
  def set_placeholders(self, indices, feed_dict={}):
    """"""

    unique_indices, inverse_indices = np.unique(indices, return_inverse=True)
    feed_dict[self.placeholder] = inverse_indices.reshape(indices.shape)
    self._multibucket.set_placeholders(unique_indices, feed_dict=feed_dict)
    return feed_dict

  #=============================================================
  def open(self):
    """"""

    self._multibucket.open()
    return self

  #=============================================================
  def close(self):
    """"""

    self._multibucket.close()
    return

  #=============================================================
  def reset(self):
    """"""

    self._idx2tok = {}
    self._tok2idx = {}
    self._multibucket.reset()

  #=============================================================
  @property
  def token_vocab_savename(self):
    return os.path.join(self.save_dir, self.field+'-tokens.lst')
  @property
  def token_vocab_loadname(self):
    return self._config.getstr(self, 'token_vocab_loadname')
  @property
  def max_buckets(self):
    return self._config.getint(self, 'max_buckets')
  @property
  def embed_keep_prob(self):
    return self._config.getfloat(self, 'embed_keep_prob')
  @property
  def conv_keep_prob(self):
    return self._config.getfloat(self, 'conv_keep_prob')
  @property
  def recur_keep_prob(self):
    return self._config.getfloat(self, 'recur_keep_prob')
  @property
  def linear_keep_prob(self):
    return self._config.getfloat(self, 'linear_keep_prob')
  @property
  def output_keep_prob(self):
    return self._config.getfloat(self, 'output_keep_prob')
  @property
  def n_layers(self):
    return self._config.getint(self, 'n_layers')
  @property
  def first_layer_conv_width(self):
    return self._config.getint(self, 'first_layer_conv_width')
  @property
  def conv_width(self):
    return self._config.getint(self, 'conv_width')
  @property
  def embed_size(self):
    return self._config.getint(self, 'embed_size')
  @property
  def recur_size(self):
    return self._config.getint(self, 'recur_size')
  @property
  def output_size(self):
    return self._config.getint(self, 'output_size')
  @property
  def hidden_size(self):
    return self._config.getint(self, 'hidden_size')
  @property
  def bidirectional(self):
    return self._config.getboolean(self, 'bidirectional')
  @property
  def drop_func(self):
    drop_func = self._config.getstr(self, 'drop_func')
    if hasattr(embeddings, drop_func):
      return getattr(embeddings, drop_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(embeddings.__name__, drop_func))
  @property
  def recur_func(self):
    recur_func = self._config.getstr(self, 'recur_func')
    if hasattr(nonlin, recur_func):
      return getattr(nonlin, recur_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, recur_func))
  @property
  def highway_func(self):
    highway_func = self._config.getstr(self, 'highway_func')
    if hasattr(nonlin, highway_func):
      return getattr(nonlin, highway_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, highway_func))
  @property
  def output_func(self):
    output_func = self._config.getstr(self, 'output_func')
    if hasattr(nonlin, output_func):
      return getattr(nonlin, output_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, output_func))
  @property
  def recur_cell(self):
    recur_cell = self._config.getstr(self, 'recur_cell')
    if hasattr(recurrent, recur_cell):
      return getattr(recurrent, recur_cell)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(recurrent.__name__, recur_func))
  @property
  def drop_type(self):
    return self._config.getstr(self, 'drop_type')
  @property
  def bilin(self):
    return self._config.getboolean(self, 'bilin')
  @property
  def cifg(self):
    return self._config.getboolean(self, 'cifg')
  @property
  def highway(self):
    return self._config.getboolean(self, 'highway')
  @property
  def squeeze_type(self):
    return self._config.getstr(self, 'squeeze_type')

#***************************************************************
class GraphSubtokenVocab(SubtokenVocab):
  """"""

  def _collect_tokens(self, node):
    node = node.split('|')
    for edge in node:
      edge = edge.split(':', 1)
      head, rel = edge
      self.counts.update(rel)

#***************************************************************
class FormSubtokenVocab(SubtokenVocab, cv.FormVocab):
  pass
class LemmaSubtokenVocab(SubtokenVocab, cv.LemmaVocab):
  pass
class UPOSSubtokenVocab(SubtokenVocab, cv.UPOSVocab):
  pass
class XPOSSubtokenVocab(SubtokenVocab, cv.XPOSVocab):
  pass
class DeprelSubtokenVocab(SubtokenVocab, cv.DeprelVocab):
  pass
