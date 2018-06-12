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
import numpy.linalg as la
import matplotlib.pyplot as plt
def pretty_plot(ax):
  ax.grid(linestyle='--', axis='y')
  
from hpo.hps.base_hyperparam import BaseHyperparam

#***************************************************************
class StringHyperparam(BaseHyperparam):
  """"""
  
  #=============================================================
  def __init__(self, section, option, bounds):
    """"""
    
    super(StringHyperparam, self).__init__(section, option, bounds)
    self._idx2str = {}
    for i, bound in enumerate(self.bounds):
      self._idx2str[i] = bound
    self._str2idx = {v: k for k, v in self._idx2str.items()}
    return
  
  #=============================================================
  def normalize(self, value):
    return self.str2idx[value]
  
  def denormalize(self, vector):
    vector = np.append(vector, 1/len(self.bounds))
    return self.idx2str[np.argmax(vector)]
  
  #=============================================================
  def get_config_value(self, config):
    return config.get(self.section, self.option)
  
  #=============================================================
  def plot(self, scores):
    """"""
    
    d = len(self.str2idx)
    fig, ax = plt.subplots()
    ax.set_title(self.section)
    ax.set_ylabel('Normalized LAS')
    ax.set_xlabel(self.option)
    x = []
    for i in six.moves.range(d):
      x.append(scores[np.where(np.equal(self.values, i))])
    ax.violinplot(x, np.arange(d), showmeans=True)
    ax.scatter(self.values, scores, alpha=.5, edgecolor='k')
    ax.set_xticks(np.arange(d))
    ax.set_xticklabels(sorted(self.str2idx, key=self.str2idx.get))
    pretty_plot(ax)
    plt.show()
    return
  
  #=============================================================
  def _rand(self):
    return np.random.choice(self.bounds)
  
  #=============================================================
  def PPP_volume(self, choice, d):
    """"""
    
    n = len(self.values)
    norm_choice = self.normalize(choice)
    if 1/np.power(n+1, 1/d) > .5*(1+1/len(self.bounds)):
      length = 1
      cluster = set(np.arange(n))
    else:
      length = 1/len(self.bounds)
      cluster = set(np.where(np.equal(self.values, norm_choice))[0])
    return length, cluster
  
  #=============================================================
  def as_matrix(self):
    """"""
    
    mat = []
    for value in self.values:
      vec = np.zeros(len(self.str2idx))
      if value is not np.nan:
        vec[value] = 1
      mat.append(vec)
    return np.array(mat)
  
  #=============================================================
  @property
  def str2idx(self):
    return self._str2idx
  @property
  def idx2str(self):
    return self._idx2str
  
