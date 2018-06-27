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

import os
import codecs
import re
from collections import Counter, OrderedDict
from collections import defaultdict as DefaultDict

import numpy as np
import tensorflow as tf

from parser.structs.vocabs.base_vocabs import BaseVocab
from . import conllu_vocabs as cv

from parser.neural import nn, nonlin, embeddings, classifiers

#***************************************************************
class FeatureVocab(BaseVocab):
  """"""
  
  _save_str = 'feats'
  
  #=============================================================
  def __init__(self, *args, placeholder_shape=[None,None,None], **kwargs):
    """"""
    
    super(FeatureVocab, self).__init__(*args, placeholder_shape=placeholder_shape, **kwargs)
    
    self._counts = DefaultDict(Counter)
    self._str2idx = DefaultDict(dict)
    self._idx2str = DefaultDict(dict)
    self.PAD_STR = self.UNK_STR = self.pad_str
    self.PAD_IDX = self.UNK_IDX = 0
    if self.keyed:
      self.ROOT_STR = 'Yes'
      self.ROOT_IDX = 1
      self._feats = ['Root']
      self._feat_set = {'Root'}
      self['Root', self.PAD_STR] = self.PAD_IDX
      self['Root', self.ROOT_STR] = self.ROOT_IDX
    else:
      self.ROOT_STR = self.pad_str
      self.ROOT_IDX = 0
      self._feats = list()
      self._feat_set = dict()
    
  #=============================================================
  def get_input_tensor(self, embed_keep_prob=None, nonzero_init=True, variable_scope=None, reuse=True):
    """"""
    
    embed_keep_prob = 1 if reuse else (embed_keep_prob or self.embed_keep_prob)
    
    layers = []
    with tf.variable_scope(variable_scope or self.classname):
      for i, feat in enumerate(self._feats):
        vs_feat = str(feat).replace('[', '-RSB-').replace(']', '-LSB-')
        with tf.variable_scope(vs_feat):
          layer = embeddings.token_embedding_lookup(self.getlen(feat), self.embed_size,
                                                    self.placeholder[:,:,i],
                                                    nonzero_init=nonzero_init,
                                                    reuse=reuse)
          layers.append(layer)
      layer = tf.add_n(layers)
      if embed_keep_prob < 1:
        layer = self.drop_func(layer, embed_keep_prob)
    return layer
  
  #=============================================================
  # TODO fix this to compute feature-level F1 rather than token-level accuracy
  def get_linear_classifier(self, layer, token_weights, last_output=None, variable_scope=None, reuse=False):
    """"""
    
    if last_output:
      n_layers = 0
      layer = last_output['hidden_layer']
      recur_layer = last_output['recur_layer']
    else:
      n_layers = self.n_layers
      recur_layer = layer
    hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
    with tf.variable_scope(variable_scope or self.classname):
      for i in six.moves.range(0, self.n_layers):
        with tf.variable_scope('FC-%d' % i):
          layer = classifiers.hidden(layer, self.hidden_size,
                                    hidden_func=self.hidden_func,
                                    hidden_keep_prob=hidden_keep_prob)
      with tf.variable_scope('Classifier'):
        probabilities = []
        loss = []
        predictions = []
        correct_tokens = []
        for i, feat in enumerate(self._feats):
          vs_feat = str(feat).replace('[', '-RSB-').replace(']', '-LSB-')
          with tf.variable_scope(vs_feat):
            logits = classifiers.linear_classifier(layer, self.getlen(feat), hidden_keep_prob=hidden_keep_prob)
            targets = self.placeholder[:,:,i]
            
            #---------------------------------------------------
            # Compute probabilities/cross entropy
            # (n x m x c) -> (n x m x c)
            probabilities.append(tf.nn.softmax(logits))
            # (n x m), (n x m x c), (n x m) -> ()
            loss.append(tf.losses.sparse_softmax_cross_entropy(targets, logits, weights=token_weights))
            
            #---------------------------------------------------
            # Compute predictions/accuracy
            # (n x m x c) -> (n x m)
            predictions.append(tf.argmax(logits, axis=-1, output_type=tf.int32))
            # (n x m) (*) (n x m) -> (n x m)
            correct_tokens.append(nn.equal(targets, predictions[-1]))
        # (n x m) x f -> (n x m x f)
        predictions = tf.stack(predictions, axis=-1)
        # (n x m) x f -> (n x m x f)
        correct_tokens = tf.stack(correct_tokens, axis=-1)
        # (n x m x f) -> (n x m)
        correct_tokens = tf.reduce_prod(correct_tokens, axis=-1) * token_weights
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
    outputs['targets'] = self.placeholder
    outputs['probabilities'] = probabilities
    outputs['loss'] = tf.add_n(loss)
    
    outputs['predictions'] = predictions
    outputs['n_correct_tokens'] = tf.reduce_sum(correct_tokens)
    outputs['n_correct_sequences'] = tf.reduce_sum(correct_sequences)
    return outputs
  
  #=============================================================
  # TODO fix this to compute feature-level F1 rather than token-level accuracy
  def get_bilinear_classifier_with_embeddings(self, layer, embeddings, token_weights, last_output=None, variable_scope=None, reuse=False):
    """"""
    
    recur_layer = layer
    hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
    with tf.variable_scope(variable_scope or self.classname):
      for i in six.moves.range(0, self.n_layers):
        with tf.variable_scope('FC-%d' % i):
          layer = classifiers.hidden(layer, self.hidden_size,
                                    hidden_func=self.hidden_func,
                                    hidden_keep_prob=hidden_keep_prob)
      with tf.variable_scope('Classifier'):
        probabilities = []
        loss = []
        predictions = []
        correct_tokens = []
        for i, feat in enumerate(self._feats):
          vs_feat = str(feat).replace('[', '-RSB-').replace(']', '-LSB-')
          with tf.variable_scope(vs_feat):
            logits = classifiers.batch_bilinear_classifier(
              layer, embeddings, self.getlen(feat),
              hidden_keep_prob=hidden_keep_prob,
              add_linear=self.add_linear)
            targets = self.placeholder[:,:,i]
            
            #---------------------------------------------------
            # Compute probabilities/cross entropy
            # (n x m x c) -> (n x m x c)
            probabilities.append(tf.nn.softmax(logits))
            # (n x m), (n x m x c), (n x m) -> ()
            loss.append(tf.losses.sparse_softmax_cross_entropy(targets, logits, weights=token_weights))
            
            #---------------------------------------------------
            # Compute predictions/accuracy
            # (n x m x c) -> (n x m)
            predictions.append(tf.argmax(logits, axis=-1, output_type=tf.int32))
            # (n x m) (*) (n x m) -> (n x m)
            correct_tokens.append(nn.equal(targets, predictions[-1]))
        # (n x m) x f -> (n x m x f)
        predictions = tf.stack(predictions, axis=-1)
        # (n x m) x f -> (n x m x f)
        correct_tokens = tf.stack(correct_tokens, axis=-1)
        # (n x m x f) -> (n x m)
        correct_tokens = tf.reduce_prod(correct_tokens, axis=-1) * token_weights
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
    outputs['targets'] = self.placeholder
    outputs['probabilities'] = probabilities
    outputs['loss'] = tf.add_n(loss)
    
    outputs['predictions'] = predictions
    outputs['n_correct_tokens'] = tf.reduce_sum(correct_tokens)
    outputs['n_correct_sequences'] = tf.reduce_sum(correct_sequences)
    return outputs
  
  #=============================================================
  # TODO finish this
  def get_bilinear_classifier(self, layer, outputs, token_weights, variable_scope=None, reuse=False):
    """"""
    
    recur_layer = layer
    hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
    with tf.variable_scope(variable_scope or self.classname):
      with tf.variable_scope(variable_scope or self.classname):
        for i in six.moves.range(0, self.n_layers-1):
          with tf.variable_scope('FC-%d' % i):
            layer = classifiers.hidden(layer, 2*hidden_size,
                                       hidden_func=self.hidden_func,
                                       hidden_keep_prob=hidden_keep_prob)
          with tf.variable_scope('FC-top'):
            layers = classifiers.hiddens(layer, 2*[hidden_size],
                                         hidden_func=self.hidden_func,
                                         hidden_keep_prob=hidden_keep_prob)
      layer1, layer2 = layers.pop(0), layers.pop(0)
      with tf.variable_scope('Classifier'):
        probabilities = []
        loss = []
        predictions = []
        correct_tokens = []
        for i, feat in enumerate(self._feats):
          vs_feat = str(feat).replace('[', '-RSB-').replace(']', '-LSB-')
          with tf.variable_scope(vs_feat):
            if self.diagonal:
              logits = classifiers.diagonal_bilinear_classifier(layer1, layer2, self.getlen(feat),
                                                                hidden_keep_prob=hidden_keep_prob,
                                                                add_linear=self.add_linear)
            else:
              logits = classifiers.bilinear_classifier(layer1, layer2, self.getlen(feat),
                                                       hidden_keep_prob=hidden_keep_prob,
                                                       add_linear=self.add_linear)
            targets = self.placeholder[:,:,i]
            
            #---------------------------------------------------
            # Compute probabilities/cross entropy
            # (n x m x c) -> (n x m x c)
            probabilities.append(tf.nn.softmax(logits))
            # (n x m), (n x m x c), (n x m) -> ()
            loss.append(tf.losses.sparse_softmax_cross_entropy(targets, logits, weights=token_weights))
            
            #---------------------------------------------------
            # Compute predictions/accuracy
            # (n x m x c) -> (n x m)
            predictions.append(tf.argmax(logits, axis=-1, output_type=tf.int32))
            # (n x m) (*) (n x m) -> (n x m)
            correct_tokens.append(nn.equal(targets, predictions[-1]))
        # (n x m) x f -> (n x m x f)
        predictions = tf.stack(predictions, axis=-1)
        # (n x m) x f -> (n x m x f)
        correct_tokens = tf.stack(correct_tokens, axis=-1)
        # (n x m x f) -> (n x m)
        correct_tokens = tf.reduce_prod(correct_tokens, axis=-1) * token_weights
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
    outputs['targets'] = self.placeholder
    outputs['probabilities'] = probabilities
    outputs['loss'] = tf.add_n(loss)
    
    outputs['predictions'] = predictions
    outputs['n_correct_tokens'] = tf.reduce_sum(correct_tokens)
    outputs['n_correct_sequences'] = tf.reduce_sum(correct_sequences)
    return outputs
  
  #=============================================================
  def getlen(self, feat):
    """"""
    
    return len(self._str2idx[feat])
  
  #=============================================================
  # TODO make this compatible with zipped files
  def count(self, train_conllus):
    """"""
    
    for train_conllu in train_conllus:
      with codecs.open(train_conllu, encoding='utf-8', errors='ignore') as f:
        for line in f:
          line = line.strip()
          if line and not line.startswith('#'):
            line = line.split('\t')
            multitoken = line[self.conllu_idx] # conllu_idx is provided by the CoNLLUVocab
            self._count(multitoken)
    self.index_by_counts()
    return True
  
  def _count(self, multitoken):
    if not self.cased:
      multitoken = multitoken.lower()
    multitoken = multitoken.split(self.separator)
    for i, token in enumerate(multitoken):
      if token != '_':
        if self.keyed:
          feat, token = token.split('=')
        else:
          feat = str(i)
        
        if feat not in self._feat_set:
          self._feats.append(feat)
          self._feat_set.add(feat)
        self._counts[feat][token] += 1
    return
  
  #=============================================================
  def add(self, token):
    """"""
    
    return self.index(token)
  
  #=============================================================
  def token(self, index):
    """"""
    
    assert isinstance(index[0], six.integer_types + (np.int32, np.int64))
    return self[index]
  
  #=============================================================
  def index(self, multitoken):
    """"""
    
    assert isinstance(multitoken, six.string_types), 'FeatureVocab.index was passed {}'.format(multitoken)
    return self[multitoken]
  
  #=============================================================
  def get_root(self):
    """"""
    
    return 'Root=Yes' if self.keyed else self.separator.join([self.ROOT_STR for _ in self._feats])
  
  #=============================================================
  @staticmethod
  def sorted(counter):
    return sorted(counter.most_common(), key=lambda x: (-x[1], x[0]))
  
  #=============================================================
  def index_by_counts(self, dump=True):
    """"""
    
    for feat, counter in six.iteritems(self._counts):
      self[feat, self.PAD_STR] = 0
      cur_idx = 1
      for token, count in self.sorted(counter):
        if (not self.min_occur_count or count >= self.min_occur_count) and\
           (not self.max_embed_count or cur_idx < self.max_embed_count+1):
          self[feat, token] = cur_idx
          cur_idx += 1
    self._depth = len(self)
    if dump:
      self.dump()
    return
  
  #=============================================================
  def dump(self):
    """"""
    
    with codecs.open(self.vocab_savename, 'w', encoding='utf-8', errors='ignore') as f:
      for feat, counter in six.iteritems(self._counts):
        f.write(u'[{}]\n'.format(feat))
        for token, count in self.sorted(counter):
          f.write(u'{}\t{}\n'.format(token, count))
        f.write(u'\n')
    return
  
  #=============================================================
  def load(self):
    """"""
    
    # First check to see if it's saved in the save_dir, then somewhere else
    dump = None
    if os.path.exists(self.vocab_savename):
      vocab_filename = self.vocab_savename
      dump = False
    elif self.vocab_loadname and os.path.exists(self.vocab_loadname):
      vocab_filename = self.vocab_loadname
      dump = True
    else:
     self._loaded = False
     return False
   
    with codecs.open(vocab_filename, encoding='utf-8', errors='ignore') as f:
      for line in f:
        line = line.rstrip()
        if line:
          featmatch = re.match('\[(.*)\]$', line) # matches '[feature]'
          match = re.match('(.*)\s([0-9]+)', line) # matches 'value count'
          if featmatch:
            feat = featmatch.group(1)
            self._feats.append(feat)
            self._counts[feat] = Counter()
          elif match:
            token = match.group(1)
            count = int(match.group(2))
            self._counts[feat][token] = count
    self.index_by_counts(dump=dump)
    self._loaded = True
    return True
  
  #=============================================================
  def __getitem__(self, key):
    assert hasattr(key, '__iter__'), 'You gave FeatureVocab.__getitem__ {}'.format(key)
    if isinstance(key, six.string_types):
      if not self.cased:
        key = key.lower()
      if key == '_':
        return [self.PAD_IDX for _ in self._feats]
      multitoken = key.split(self.separator)
      if self.keyed:
        key_dict = {}
        for token in multitoken:
          feat, token = token.split('=')
          key_dict[feat] = token
        return [self._str2idx[feat].get(key_dict[feat], self.UNK_IDX) if feat in key_dict else self.PAD_IDX for feat in self._feats]
      else:
        return [self._str2idx[str(feat)].get(key, self.UNK_IDX) for feat, key in enumerate(multitoken)]
    elif isinstance(key[0], six.integer_types + (np.int32, np.int64)):
      if self.keyed:
        multitoken = ['{}={}'.format(feat, self._idx2str[feat].get(key, self.UNK_STR)) for feat, key in zip(self._feats, key) if key != self.PAD_IDX]
      else:
        multitoken = [self._idx2str[feat].get(key, self.UNK_STR) for feat, key in enumerate(key)]
      return self.separator.join(multitoken) or '_'
    else:
      return [self[k] for k in key]
  
  def __setitem__(self, key, value):
    if len(key) == 1:
      assert len(value) == 1
      self[key[0]] = value[0]
      return 
    
    if isinstance(key[1], six.string_types):
      vocab, key = key
      if not self.cased and key != self.PAD_STR:
        key = key.lower()
      self._str2idx[vocab][key] = value
      self._idx2str[vocab][value] = key
    elif isinstance(key, six.integer_types + (np.int32, np.int64)):
      vocab, key = key
      if not self.cased and value != self.PAD_STR:
        value = value.lower()
      self._idx2str[vocab][key] = value
      self._str2idx[vocab][value] = key
    elif hasattr(key, '__iter__') and hasattr(value, '__iter__'):
      for k, v in zip(key, value):
        self[k] = v
    else:
      raise ValueError('keys and values to {}.__setitem__ must be (iterables of) (string, string or integer) tuples and string or integer')
  
  def __contains__(self, key):
    assert isinstance(key, (tuple, list))
    
    vocab, key = key
    if isinstance(key, six.string_types):
      if not self.cased and key != self.PAD_STR:
        key = key.lower()
      return key in self._str2idx[vocab]
    elif isinstance(key, six.integer_types + (np.int32, np.int64)):
      return key in self._idx2str[vocab]
    else:
      raise ValueError('key to {}.__contains__ must be (string, string or integer) tuple')
  
  def __len__(self):
    return len(self._str2idx)
  
  def __iter__(self):
    return (vocab for vocab in self._str2idx)
  
  #=============================================================
  @property
  def drop_func(self):
    drop_func = self._config.getstr(self, 'drop_func')
    if hasattr(embeddings, drop_func):
      return getattr(embeddings, drop_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(embeddings.__name__, drop_func))
  @property
  def embed_size(self):
    return self._config.getint(self, 'embed_size')
  @property
  def embed_keep_prob(self):
    return self._config.getfloat(self, 'embed_keep_prob')
  @property
  def hidden_func(self):
    hidden_func = self._config.getstr(self, 'hidden_func')
    if hasattr(nonlin, hidden_func):
      return getattr(nonlin, hidden_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, hidden_func))
  @property
  def hidden_size(self):
    return self._config.getint(self, 'hidden_size')
  @property
  def add_linear(self):
    return self._config.getboolean(self, 'add_linear')
  @property
  def n_layers(self):
    return self._config.getint(self, 'n_layers')
  @property
  def hidden_keep_prob(self):
    return self._config.getfloat(self, 'hidden_keep_prob')
  @property
  def max_embed_count(self):
    return self._config.getint(self, 'max_embed_count')
  @property
  def min_occur_count(self):
    return self._config.getint(self, 'min_occur_count')
  @property
  def vocab_loadname(self):
    return self._config.getstr(self, 'vocab_loadname')
  @property
  def vocab_savename(self):
    return os.path.join(self.save_dir, self.field+'-'+self._save_str+'.lst')
  @property
  def keyed(self):
    return self._config.getboolean(self, 'keyed')
  @property
  def separator(self):
    separator = self._config.getstr(self, 'separator')
    if separator is None:
      return ''
    else:
      return separator
  @property
  def pad_str(self):
    pad_str = self._config.getstr(self, 'pad_str')
    if pad_str is None:
      pad_str = ''
    return pad_str
  @property
  def cased(self):
    return self._config.getboolean(self, 'cased')

#***************************************************************  
class LemmaFeatureVocab(FeatureVocab, cv.LemmaVocab):
  pass
class XPOSFeatureVocab(FeatureVocab, cv.XPOSVocab):
  pass
class UFeatsFeatureVocab(FeatureVocab, cv.UFeatsVocab):
  pass
