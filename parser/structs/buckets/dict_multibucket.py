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

from .base_multibucket import BaseMultibucket
from .dict_bucket import DictBucket
 
#***************************************************************
class DictMultibucket(BaseMultibucket, dict):
  """"""
  
  #=============================================================
  def __init__(self, vocabs, max_buckets=2, config=None):
    """"""
    
    super(DictMultibucket, self).__init__(max_buckets, config=config)
    dict.__init__(self)
    
    for vocab in vocabs:
      self[vocab.classname] = [DictBucket(idx, vocab.depth, config=config) for idx in six.moves.range(max_buckets)]
    
    self._lengths = []
    self._indices = {vocab.classname: [] for vocab in vocabs}
    self._tokens = {vocab.classname: [] for vocab in vocabs}
    self._file_indices = []
    self._max_lengths = []
    return
  
  #=============================================================
  def reset(self):
    """"""
    
    self._lengths = []
    self._indices = {vocab.classname: [] for vocab in vocabs}
    self._tokens = {vocab.classname: [] for vocab in vocabs}
    self._file_indices = []
    for vocab_classname in self:
      for bucket in self[vocab_classname]:
        bucket.reset()
    return
  
  #=============================================================
  def add(self, indices, tokens, file_index=-1, length=0):
    """"""
    
    assert self._is_open, 'The DictMultibucket is not open for adding entries'
    
    if length <= 1:
      return
    for vocab_classname in indices.keys():
      self._indices[vocab_classname].append(indices[vocab_classname])
    for vocab_classname in tokens.keys():
      self._tokens[vocab_classname].append(tokens[vocab_classname])
    self._file_indices.append(file_index)
    super(DictMultibucket, self).add(length)
    return 
  
  #=============================================================
  def close(self):
    """"""
    
    # Decide where everything goes
    self._max_lengths = self.compute_max_lengths(self._lengths, self.max_buckets)
    len2bkt = self.get_len2bkt(self._max_lengths)
    
    # Open the buckets
    shape = len(self._lengths)
    dtype = [('file', 'i4'), ('bucket', 'i4')] + [(vocab_classname, 'i4') for vocab_classname in self.keys()]
    data = np.zeros(shape, dtype=dtype)
    for i, vocab_classname in enumerate(self.keys()):
      for bucket in self[vocab_classname]:
        bucket.open()
    
      # Add sentences to them
      for j, (indices, tokens) in enumerate(zip(self._indices[vocab_classname], self._tokens[vocab_classname])):
        bucket_index = len2bkt[len(indices)]
        sequence_index = self[vocab_classname][bucket_index].add(indices, tokens)
        data[vocab_classname][j] = sequence_index
        if i == 0:
          data['file'][j] = self._file_indices[j]
          data['bucket'][j] = bucket_index
        else:
          assert data['bucket'][j] == bucket_index, 'CoNLLU data is somehow misaligned'
    
    # Close the buckets
    for vocab_classname in self:
      for bucket in self[vocab_classname]:
        bucket.close()
    super(DictMultibucket, self).close(data)
    
    return
  
  #=============================================================
  def get_data(self, vocab_classname, indices):
    """"""
    
    bucket_index = np.unique(self.bucket_indices[indices])
    assert len(bucket_index) == 1, 'Requested data from multiple (or no) buckets'
    
    bucket_index = bucket_index[0]
    data_indices = self.data[vocab_classname][indices]
    data = self[vocab_classname][bucket_index].data[data_indices]
    return data
  
  #=============================================================
  def get_tokens(self, vocab_classname, indices):
    """"""
    
    return [self._tokens[vocab_classname][index] for index in indices]
  
  #=============================================================
  @property
  def max_lengths(self):
    return self._max_lengths
  @property
  def file_indices(self):
    return self.data['file']
  @property
  def bucket_indices(self):
    return self.data['bucket']
  @property
  def data_indices(self):
    return self.data[:, 2:]
