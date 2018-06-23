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
from collections import Counter, OrderedDict
from collections import defaultdict as DefaultDict

import numpy as np
import tensorflow as tf

from parser.structs.vocabs.base_vocabs import SetVocab
from . import conllu_vocabs as cv

from parser.neural import nn, nonlin, embeddings, classifiers

#***************************************************************
class FeatureVocab(BaseVocab):
  """"""
  
  PAD_STR = ROOT_STR = UNK_STR = self.pad_str
  PAD_IDX = ROOT_IDX = UNK_IDX = 0
  _save_str = 'feats'
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    super(FeatureVocab, self).__init__(*args, **kwargs)
    
    self._counts = DefaultDict(Counter)
    self._keys = list()
    self._key_set = set()
    self._str2idx = DefaultDict(dict)
    self._idx2str = DefaultDict(dict)
    
  #=============================================================
  def get_input_tensor(self, embed_keep_prob=None, nonzero_init=True, variable_scope=None, reuse=True):
    """"""
    
    embed_keep_prob = 1 if reuse else (embed_keep_prob or self.embed_keep_prob)
    
    layers = []
    with tf.variable_scope(variable_scope or self.classname):
      for i, vocab in enumerate(self):
        with tf.variable_scope(str(vocab)):
          layer = embeddings.token_embedding_lookup(self.len(vocab), self.embed_size,
                                                    self.placeholder[i],
                                                    nonzero_init=nonzero_init,
                                                    reuse=reuse)
      layer = tf.add_n(layers)
      if embed_keep_prob < 1:
        layer = self.drop_func(layer, embed_keep_prob)
    return layer
  
  #=============================================================
  # TODO fix this to compute feature-level F1 rather than token-level accuracy
  def get_linear_classifier(self, layer, token_weights, variable_scope=None, reuse=False):
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
        for i, key in enumerate(self._keys):
          with tf.variable_scope(key):
            logits = classifiers.linear_classifier(layer, self.getlen(key), hidden_keep_prob=hidden_keep_prob)
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
            correct_tokens.append(nn.equal(targets, predictions))
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
  def getlen(self, key):
    """"""
    
    return len(self._str2idx[key])
  
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
      if self.keyed:
        key, token = token.split('=')
      else:
        key = str(i)
      
      if key not in self._key_set:
        self._keys.append(key)
        self._key_set.add(key)
      self._counts[key][token] += 1
    return
  
  #=============================================================
  def add(self, token):
    """"""
    
    return self.index(token)
  
  #=============================================================
  def token(self, index):
    """"""
    
    assert len(self._keys) == len(index)
    return self.separator.join([self[key, idx] for key, idx in zip(self._keys, index)])
  
  #=============================================================
  def index(self, token):
    """"""
    
    if not self.cased:
      multitoken = multitoken.lower()
    multitoken = multitoken.split(self.separator)
    if self.keyed:
      key_dict = {}
      for token in multitoken:
        key, token = token.split('=')
        key_dict[key] = token
      return [self[key, key_dict.get(key, self.PAD_STR)] for key in self._keys]
    else:
      return [self[str(key), token] for key, token in enumerate(multitoken)]
  
  #=============================================================
  def get_root(self):
    """"""
    
    return [(vocab, self.ROOT_STR) for vocab in self]
  
  #=============================================================
  @staticmethod
  def sorted(counter):
    return sorted(counter.most_common(), key=lambda x: (-x[1], x[0]))
  
  #=============================================================
  def index_by_counts(self, dump=True):
    """"""
    
    for key, counter in six.moves.iteritems(self._counts):
      self[key, self.PAD_STR] = 0
      cur_idx = 1
      for token, count in self.sorted(counter):
        if (not self.min_occur_count or count >= self.min_occur_count) and\
           (not self.max_embed_count or cur_idx < self.max_embed_count+1):
          self[key, token] = cur_idx
          cur_idx += 1
    self._depth = len(self)
    if dump:
      self.dump()
    return
  
  #=============================================================
  def dump(self):
    """"""
    
    with codecs.open(self.vocab_savename, 'w', encoding='utf-8', errors='ignore') as f:
      for key, counter in six.moves.iteritems(self._counts):
        f.write(u'[{}]\n'.format(key))
        for token, count in self.sorted(counter):
          f.write(u'{}\t{}\n'.format(token, count))
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
     return False
   
    with codecs.open(vocab_filename, encoding='utf-8', errors='ignore') as f:
      for line in f:
        line = line.rstrip()
        key = None
        if line:
          keymatch = re.match('\[(.*)\]$', line)
          match = re.match('(.*)\s([0-9]*)', line)
          if keymatch:
            key = keymatch.group(1)
            self._counts[key] = Counter()
          elif match:
            token = match.group(1)
            count = int(match.group(2))
            self._counts[key][token] = count
    self.index_by_counts(dump=dump)
    return True
  
  #=============================================================
  def __getitem__(self, key):
    if not key:
      return []
    if len(key) == 1:
      return [self[key[0]]]
    
    if isinstance(key[1], six.string_types):
      vocab, key = key
      if not self.cased and key != self.PAD_STR:
        key = key.lower()
      return self._str2idx[vocab].get(key, self.UNK_STR)
    elif isinstance(key, six.integer_types + (np.int32, np.int64)):
      vocab, key = key
      if self.keyed:
        token = self._idx2str[vocab].get(key, self.UNK_IDX)
        return '{}={}'.format(vocab, token)
      else:
        return token
    elif hasattr(key, '__iter__'):
      return [self[k] for k in key]
    else:
      raise ValueError('key to {}.__getitem__ must be (iterable of) (string, string or integer) tuples'.format(self.classname))
  
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
    return self._config.getstr(self, 'pad_str')
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
