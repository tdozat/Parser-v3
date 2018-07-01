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

from .base_multibucket import BaseMultibucket
from .list_bucket import ListBucket
 
#***************************************************************
class ListMultibucket(BaseMultibucket, list):
  """"""
  
  #=============================================================
  def __init__(self, vocab, max_buckets=3, config=None):
    """"""
    
    super(ListMultibucket, self).__init__(max_buckets, config=config)
    list.__init__(self)
    
    for _ in six.moves.range(max_buckets):
      self.append(ListBucket(len(self), config=config))
    
    self._vocab_classname = vocab.classname
    self._lengths = [0]
    self._indices = [[]]
    self._tokens = [[]]
    self._str2idx = {'': 0}
    self.placeholder = tf.placeholder(tf.int32, [None], name=self.classname)
    return
  
  #=============================================================
  def reset(self):
    """"""
    
    self._lengths = [0]
    self._indices = [[]]
    self._tokens = [[]]
    self._str2idx = {'': 0}
    return
  
  #=============================================================
  def get_placeholders(self):
    """"""
    
    for bucket in self:
      yield bucket.placeholder
  
  #=============================================================
  def add(self, indices, tokens):
    """"""
    
    assert self._is_open, 'Multibucket is not open for adding entries'
    
    string = ' '.join(tokens)
    if string in self._str2idx:
      index = self._str2idx[string]
    else:
      index = len(self._indices)
      self._indices.append(indices)
      self._tokens.append(tokens)
      self._str2idx[string] = index
      super(ListMultibucket, self).add(len(indices))
    
    return index
  
  #=============================================================
  def close(self):
    """"""
    
    # Decide where everything goes
    max_lengths = self.compute_max_lengths(self._lengths, self.max_buckets)
    len2bkt = self.get_len2bkt(max_lengths)
    
    # Open the buckets
    shape = len(self._lengths)
    dtype=[('bucket', 'i4'), ('sequence', 'i4'), ('length', 'i4')]
    data = np.zeros(shape, dtype=dtype)
    for bucket in self:
      bucket.open()
    
    # Add sentences to them
    for i, (indices, length) in enumerate(zip(self._indices, self._lengths)):
      bucket_index = len2bkt[len(indices)]
      sequence_index = self[bucket_index].add(indices)
      data[i] = (bucket_index, sequence_index, length)
    
    # Close the buckets
    for bucket in self:
      bucket.close()
    super(ListMultibucket, self).close(data)
    
    return
  
  #=============================================================
  def set_placeholders(self, indices, feed_dict={}):
    """"""
    
    subdata = self.data[indices]
    bucket_indices = subdata['bucket']
    bucket_sequence_indices = []
    unique_bucket_indices = set(np.unique(bucket_indices))
    for i in six.moves.range(len(self)):
      if i in unique_bucket_indices:
        bucket_i_indices = np.where(bucket_indices == i)[0]
        bucket_i_sequence_indices = subdata['sequence'][bucket_i_indices]
        maxlen = max(1, np.max(subdata['length'][bucket_i_indices]))
        self[i].set_placeholders(bucket_i_sequence_indices, maxlen, feed_dict=feed_dict)
        bucket_sequence_indices.append(bucket_i_indices)
        
    for i in six.moves.range(len(self)):
      if i not in unique_bucket_indices:
        bucket_i_indices = len(bucket_indices)+i
        bucket_i_sequence_indices = np.array([0], dtype=np.int32)
        self[i].set_placeholders(bucket_i_sequence_indices, 1, feed_dict=feed_dict)
        bucket_sequence_indices.append(bucket_i_indices)
    
    bucket_sequence_indices = np.hstack(bucket_sequence_indices)
    feed_dict[self.placeholder] = np.argsort(bucket_sequence_indices)
    
    return feed_dict
  
  #=============================================================
  @property
  def vocab_classname(self):
    return self._vocab_classname
  @property
  def tokens(self):
    return self._tokens
  @property
  def bucket_indices(self):
    return self.data['bucket']
  @property
  def sequence_indices(self):
    return self.data['sequence']
  
