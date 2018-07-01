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

from collections import Counter

import numpy as np

#***************************************************************
class BaseMultibucket(object):
  """"""
  
  #=============================================================
  def __init__(self, max_buckets, config=None):
    """"""
    
    self._max_buckets = max_buckets
    self._is_open = False
    self._data = None
    self._config = config
    return
    
  #=============================================================
  def open(self):
    """"""
    
    self._is_open = True
    self._data = None
    return self
  
  #=============================================================
  def add(self, length):
    """"""
    
    self._lengths.append(length)
    return
  
  #=============================================================
  def close(self, data):
    """"""
    
    self._is_open = False
    self._data = data
    return
  
  #=============================================================
  @property
  def data(self):
    return self._data
  @property
  def max_buckets(self):
    return self._max_buckets
  
  #=============================================================
  @staticmethod
  def get_len2bkt(max_lengths):
    """"""
    
    len2bkt = {}
    prevlen = -1
    for idx, max_length in enumerate(max_lengths):
      len2bkt.update(zip(range(prevlen+1, max_length+1), [idx]*(max_length-prevlen)))
      prevlen = max_length
    return len2bkt
  
  #=============================================================
  @staticmethod
  def compute_max_lengths(lengths, max_buckets):
    """"""
    
    uniq_lengths, count_lengths = np.unique(lengths, return_counts=True)
    length_pdf = (uniq_lengths * count_lengths) / np.sum(lengths)
    length_cdf = np.cumsum(length_pdf)
    linspace = (np.arange(1, max_buckets)) / max_buckets
    dists = (linspace[:,None] - length_cdf)**2 / 2
    all_scores = []
    for boundary, scores in enumerate(dists):
      for length, dist in enumerate(scores):
        all_scores.append((dist, boundary, length))
    all_scores.sort()
    splits = set([np.max(lengths)])
    n, m = dists.shape
    undecided_boundaries = np.ones(n)
    undecided_lengths = np.ones(m)
    for dist, boundary, length in all_scores:
      if undecided_boundaries[boundary] and undecided_lengths[length]:
        splits.add(length)
        undecided_boundaries[boundary] = 0
        undecided_lengths[length] = 0
    return sorted(list(splits))
    
  #=============================================================
  def __enter__(self):
    return self
  def __exit__(self, exception_type, exception_value, trace):
    if exception_type is not None:
      raise 
    self.close()
    return

  #=============================================================
  @property
  def classname(self):
    return self.__class__.__name__
