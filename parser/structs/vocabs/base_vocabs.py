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
from collections import Counter

import numpy as np
import tensorflow as tf

import parser.neural.nn
# id_vocab.root = 0
# form_vocab.root = <ROOT>
# lemma_vocab.root = <ROOT>
# upos_vocab.root = ROOT
# xpos_vocab.root = ROOT
# head_vocab.root = 0
# deprel_vocab.root = root
# semrel_vocab.root = root

#***************************************************************
class BaseVocab(object):
  """"""

  _depth = 0

  #=============================================================
  def __init__(self, placeholder_shape=[None, None], config=None):
    """"""

    self.placeholder = tf.placeholder(tf.int32, placeholder_shape, name=self.classname)
    self._config = config
    return

  #=============================================================
  def __call__(self):
    raise NotImplementedError('%s has no __call__' % self.classname)

  #=============================================================
  def add_sequence(self, tokens):
    """"""

    return [self.add(token) for token in tokens]

  #=============================================================
  def token_sequence(self, indices):
    """"""

    return [self.token(index) for index in indices]

  #=============================================================
  def index_sequence(self, tokens):
    """"""

    return [self.index(token) for token in tokens]

  #=============================================================
  def set_placeholders(self, indices, feed_dict={}):
    """"""

    feed_dict[self.placeholder] = indices
    return feed_dict

  #=============================================================
  def get_root(self):
    raise NotImplementedError('get_root not implemented for %s' % self.classname)

  #=============================================================
  def load(self):
    return True

  def open(self):
    return self

  def close(self):
    return

  #=============================================================
  @property
  def depth(self):
    return self._depth
  @property
  def classname(self):
    return self.__class__.__name__

  #=============================================================
  def __enter__(self):
    return self
  def __exit__(self, exception_type, exception_value, trace):
    if exception_type is not None:
      raise
    self.close()
    return
  def __hash__(self):
    return hash(self.classname)

#***************************************************************
class SetVocab(BaseVocab):
  """"""

  _base_special_tokens = [u'pad', u'root', u'unk']

  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""

    super(SetVocab, self).__init__(*args, **kwargs)
    self._cased = self._config.getboolean(self, 'cased') # cached for faster access

    # Set the special tokens
    special_tokens = [getattr(base_special_token, self._config.getstr(self, 'special_token_case'))() for base_special_token in self._base_special_tokens]
    if self._config.getboolean(self, 'special_token_html'):
      special_tokens = [u'<%s>' % special_token for special_token in special_tokens]

    # Add special tokens to the object
    for i, base_special_token in enumerate(self._base_special_tokens):
      self.__dict__[base_special_token.upper()+'_IDX'] = i
      self.__dict__[base_special_token.upper()+'_STR'] = special_tokens[i]

    # Initialize the dictionaries
    self._str2idx = dict(zip(special_tokens, range(len(special_tokens))))
    self._idx2str = dict(zip(range(len(special_tokens)), special_tokens))

    self._special_tokens = set(special_tokens)
    return

  #=============================================================
  def add(self, token):
    """"""

    return self.index(token)

  #=============================================================
  def token(self, index):
    """"""

    assert isinstance(index, six.integer_types + (np.int32, np.int64))
    return self[index]

  #=============================================================
  def index(self, token):
    """"""

    assert isinstance(token, six.string_types)
    return self[token]

  #=============================================================
  def get_root(self):
    """"""

    return self.ROOT_STR

  #=============================================================
  @property
  def cased(self):
    return self._cased
  @property
  def base_special_tokens(self):
    return self._base_special_tokens
  @property
  def special_tokens(self):
    return self._special_tokens

  #=============================================================
  def __getitem__(self, key):
    if isinstance(key, six.string_types):
      if not self.cased and key not in self.special_tokens:
        key = key.lower()
      return self._str2idx.get(key, self.UNK_IDX)
    elif isinstance(key, six.integer_types + (np.int32, np.int64)):
      return self._idx2str.get(key, self.UNK_STR)
    elif hasattr(key, '__iter__'):
      return [self[k] for k in key]
    else:
      raise ValueError('key to SetVocab.__getitem__ must be (iterable of) string or integer')
    return

  def __setitem__(self, key, value):
    if isinstance(key, six.string_types):
      if not self.cased and key not in self.special_tokens:
        key = key.lower()
      self._str2idx[key] = value
      self._idx2str[value] = key
    elif isinstance(key, six.integer_types + (np.int32, np.int64)):
      if not self.cased and value not in self.special_tokens:
        value = value.lower()
      self._idx2str[key] = value
      self._str2idx[value] = key
    elif hasattr(key, '__iter__') and hasattr(value, '__iter__'):
      for k, v in zip(key, value):
        self[k] = v
    else:
      raise ValueError('keys and values to SetVocab.__setitem__ must be (iterables of) string or integer')

  def __contains__(self, key):
    if isinstance(key, six.string_types):
      if not self.cased and key not in self.special_tokens:
        key = key.lower()
      return key in self._str2idx
    elif isinstance(key, six.integer_types + (np.int32, np.int64)):
      return key in self._idx2str
    else:
      raise ValueError('key to SetVocab.__contains__ must be string or integer')
    return

  def __len__(self):
    return len(self._str2idx)

  def __iter__(self):
    return (key for key in sorted(self._str2idx, key=self._str2idx.get))

#***************************************************************
class CountVocab(SetVocab):
  """"""

  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""

    super(CountVocab, self).__init__(*args, **kwargs)
    self._counts = Counter()
    self._min_occur_count = self._config.getint(self, 'min_occur_count')

  #=============================================================
  def index_by_counts(self, dump=True):
    """"""

    cur_idx = len(self)
    for token, count in self.sorted():
      if token not in self.special_tokens and count >= self.min_occur_count:
        self[token] = cur_idx
        cur_idx += 1
    if dump:
      self.dump()
    return

  #=============================================================
  def dump(self):
    """"""

    with codecs.open(self.filename, 'w', encoding='utf-8', errors='ignore') as f:
      for token, count in self.sorted():
        f.write(u'{}\t{}\n'.format(token, count))
    return

  #=============================================================
  def load(self):
    """"""

    cur_idx = len(self.special_tokens)
    if os.path.exists(self.filename):
      with codecs.open(self.filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
          line = line.rstrip()
          if line:
            token, count = line.split('\t')
            self.counts[token] = int(count)
      self.index_by_counts(dump=False)
      return True
    else:
      return False

  #=============================================================
  def sorted(self):
    return sorted(self.counts.most_common(), key=lambda x: (-x[1], x[0]))

  #=============================================================
  @property
  def filename(self):
    raise NotImplementedError()
  @property
  def counts(self):
    return self._counts
  @property
  def min_occur_count(self):
    return self._min_occur_count
