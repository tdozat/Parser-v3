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
class ListBucket(BaseBucket):
  """"""
  
  #=============================================================
  def __init__(self, idx, config=None):
    """"""
    
    super(ListBucket, self).__init__(idx, config=config)
    
    self._indices = [[]]
    self.placeholder = tf.placeholder(tf.int32, [None, None], name=self.__class__.__name__+'-{}'.format(self.idx))
    return
  
  #=============================================================
  def reset(self):
    """"""
    
    self._indices = [[]]
    return

  #=============================================================
  def add(self, indices):
    """"""
    
    assert self._is_open, 'ListBucket is not open for adding entries'
    
    sequence_index = len(self._indices)
    super(ListBucket, self).add(indices)
    return sequence_index
  
  #=============================================================
  def close(self):
    """"""
    
    # Initialize the index matrix
    shape = []
    shape.append(len(self._indices))
    shape.append(max(len(indices) for indices in self._indices))
    
    data = np.zeros(shape, dtype=np.int32)
    # Add data to the index matrix
    for i, sequence in enumerate(self._indices):
      if sequence:
        data[i, 0:len(sequence)] = sequence
    
    super(ListBucket, self).close(data)
    return
  
  #=============================================================
  def set_placeholders(self, indices, feed_dict={}):
    """"""
    
    feed_dict[self.placeholder] = self.data[indices]
    return feed_dict
  
