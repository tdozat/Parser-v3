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

import numpy as np
import tensorflow as tf

from .base_bucket import BaseBucket

#***************************************************************
class DictBucket(BaseBucket):
  """"""
  
  #=============================================================
  def __init__(self, idx, depth, config=None):
    """"""
    
    super(DictBucket, self).__init__(idx, config=config)
    
    self._depth = depth
    self._indices = []
    self._tokens = []
    self._str2idx = {}
    
    return
  
  #=============================================================
  def reset(self):
    """"""
    
    self._indices = []
    self._tokens = []
    self._str2idx = {}
    return

  #=============================================================
  def add(self, indices, tokens):
    """"""
    
    assert self._is_open, 'DictBucket is not open for adding entries'
    
    string = ' '.join(tokens)
    if string in self._str2idx:
      sequence_index = self._str2idx[string]
    else:
      sequence_index = len(self._indices)
      self._str2idx[string] = sequence_index
      self._tokens.append(tokens)
      super(DictBucket, self).add(indices)
    return sequence_index
  
  #=============================================================
  def close(self):
    """"""
    
    # Initialize the index matrix
    first_dim = len(self._indices)
    second_dim = max(len(indices) for indices in self._indices) if self._indices else 0
    shape = [first_dim, second_dim]
    if self.depth > 0:
      shape.append(self.depth)
    elif self.depth == -1:
      shape.append(shape[-1])
    
    data = np.zeros(shape, dtype=np.int32)
    # Add data to the index matrix
    if self.depth >= 0:
      try:
        for i, sequence in enumerate(self._indices):
          if sequence:
            data[i, 0:len(sequence)] = sequence
      except ValueError:
        print('Expected shape: {}\nsequence: {}'.format([len(sequence), self.depth], sequence))
        print(self._tokens[i])
        raise
    elif self.depth == -1:
      # for graphs, sequence should be list of (idx, val) pairs
      for i, sequence in enumerate(self._indices):
        for j, node in enumerate(sequence):
          for edge in node:
            if isinstance(edge, (tuple, list)):
              edge, v = edge
              data[i, j, edge] = v
            else:
              data[i, j, edge] = 1
    
    super(DictBucket, self).close(data)
    
    return
  
  #=============================================================
  @property
  def depth(self):
    return self._depth
  @property
  def data_indices(self):
    return self._data
