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
import scipy.stats as sps
import matplotlib.pyplot as plt
def pretty_plot(ax):
  ax.grid(linestyle='--', axis='y')
  
from hpo.hps.base_hyperparam import NumericHyperparam

#***************************************************************
class BoolHyperparam(NumericHyperparam):
  """"""
  
  #=============================================================
  def PPP_volume(self, choice, d):
    """"""
    
    n = len(self.values)
    norm_choice = self.normalize(choice)
    if np.power(n+1, -1/d) > .75:
      length = 1
      cluster = set(np.arange(n))
    else:
      length = 1/2
      cluster = set(np.where(np.equal(self.values, norm_choice))[0])
    return length, cluster
    
  #=============================================================
  def denormalize(self, value):
    return bool(np.clip(np.round(super(BoolHyperparam, self).denormalize(value)), 0, 1))
  
  #=============================================================
  def as_matrix(self):
    """"""
    
    mat = []
    mat = np.array(self.values, dtype=float)[:,None]
    mat[np.isnan(mat)] = .5
    return mat
  
  #=============================================================
  def plot(self, scores):
    """"""
    
    #-----------------------------------------------------------
    def compute_weights(scores):
      """ computes softmax(log(len(scores)) * scores) """
      
      scores = scores - np.max(scores)
      exp_scores = len(scores)**scores
      weights = exp_scores / exp_scores.sum()
      return weights
    #-----------------------------------------------------------
    weights = compute_weights(scores)
    
    x = np.array([self.denormalize(value) for value in self.values])
    minx = np.min(x)
    maxx = np.max(x)
    _range = np.linspace(-.5, 1.5)
    
    mean = weights.dot(x)
    centered = x-mean
    var = np.sqrt(centered.dot(np.diag(weights)).dot(centered)) / (1-weights.dot(weights))
    dist = sps.norm.pdf(_range, mean, np.sqrt(var))
    
    fig, ax = plt.subplots()
    ax.set_title(self.section)
    ax.set_ylabel('Normalized LAS')
    ax.set_xlabel(self.option)
    violin = []
    for i in six.moves.range(2):
      violin.append(scores[np.where(np.equal(self.values, i))])
    ax.violinplot(violin, np.arange(2), showmeans=True)
    ax.scatter(self.values, scores, alpha=.5, edgecolor='k')
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(['False', 'True'])
    axt = ax.twinx()
    axt.plot(_range, dist)
    axt.fill_between(_range, dist, alpha=.25)
    pretty_plot(ax)
    plt.show()
    return
  
  def _process_bounds(self):
    self._bounds = [(bound == 'True') for bound in self.bounds]
    return
  
  def get_config_value(self, config):
    return config.getboolean(self.section, self.option)
  
  def _rand(self):
    return np.random.choice([self.lower, self.upper])
