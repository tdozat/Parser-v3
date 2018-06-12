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

#***************************************************************
class BaseBucket(object):
  """"""
  
  #=============================================================
  def __init__(self, idx, config=None):
    """"""
    
    self._idx = idx
    
    self._is_open = False
    self._data = None
    self._config = config
  
  #=============================================================
  def open(self):
    """"""
    
    self._is_open = True
    self._data = None
    return self
  
  #=============================================================
  def add(self, indices):
    """"""
    
    self._indices.append(indices)
    return
  
  #=============================================================
  def close(self, data):
    """"""
    
    self._data = data
    self._is_open = False
    
    return
  
  #=============================================================
  @property
  def idx(self):
    return self._idx
  @property
  def data(self):
    return self._data
