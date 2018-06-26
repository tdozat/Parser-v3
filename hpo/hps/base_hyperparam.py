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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
  
def pretty_plot(ax):
  ax.grid(linestyle='--', axis='y')
 
#***************************************************************
class BaseHyperparam(object):
  """"""
  
  #=============================================================
  def __init__(self, section, option, bounds):
    """"""
    
    self._section = section
    self._option = option
    self._bounds = bounds.split(':')
    self._fixed = False
    if len(set(self.bounds)) == 1:
      self._bounds = list(set(self.bounds))
      self._fixed = True
    self._copies = [(section, option)]
    self._values = []
    return
  
  #=============================================================
  def add_copy(self, section, option):
    """"""
    
    self._copies.append( (section, option) )
    return
    
  #=============================================================
  def add_config_value(self, config):
    """"""
    
    if not config.has_section(self.section) or \
       not config.has_option(self.section, self.option) or \
       config.get(self.section, self.option) in ('', 'None'):
      self._values.append(np.nan)
    else:
      self._values.append(self.normalize(self.get_config_value(config)))
    return
  
  def normalize(self, value):
    raise NotImplementedError()
  
  def denormalize(self, value):
    raise NotImplementedError()
  
  def get_config_value(self, config):
    raise NotImplementedError()
  
  #=============================================================
  def as_matrix(self):
    """"""
    
    mat = np.array(self.values)
    mat[np.isnan(mat)] = 0
    mat = mat[:,None]
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
    _range = np.linspace(minx, maxx)
    
    mean = weights.dot(x)
    centered = x-mean
    var = centered.dot(np.diag(weights)).dot(centered) / (1-weights.dot(weights))
    dist = sps.norm.pdf(_range, mean, np.sqrt(var))
    #print(mean, var, dist)
    
    x = x[:,None]
    X = np.concatenate([np.ones_like(x), x, x**2], axis=1).astype(float)
    theta = la.inv(X.T.dot(X)+.05*np.eye(3)).dot(X.T).dot(scores)
    b, w1, w2 = theta
    curve = b + w1*_range + w2*_range**2
    optimum = -.5*w1/w2
    
    fig, ax = plt.subplots()
    ax.set_title(self.section)
    ax.set_ylabel('Normalized LAS')
    ax.set_xlabel(self.option)
    ax.scatter([self.denormalize(value) for value in self.values], scores, alpha=.5, edgecolor='k')
    ax.plot(_range, curve, color='c' if w2 < 0 else 'r')
    if optimum < maxx and optimum > minx:
      ax.axvline(optimum, ls='--', color='c' if w2 < 0 else 'r')
    axt = ax.twinx()
    axt.plot(_range, dist)
    axt.fill_between(_range, dist, alpha=.25)
    pretty_plot(ax)
    plt.show()
    return
  
  #=============================================================
  def sort(self, order):
    """"""
    
    self._values = np.array(self.values)
    self._values = self._values[order]
    return
  
  #=============================================================
  def rand(self):
    """"""
    
    if self.fixed:
      return self.bounds[0]
    else:
      return self._rand()
  
  def _rand(self):
    raise NotImplementedError() 
  
  #=============================================================
  def __hash__(self):
    return hash(self.name)
  
  #=============================================================
  @property
  def section(self):
    return self._section
  @property
  def option(self):
    return self._option
  @property
  def name(self):
    return self._section + ':' + self._option
  @property
  def bounds(self):
    return self._bounds
  @property
  def fixed(self):
    return self._fixed
  @property
  def copies(self):
    return self._copies
  @property
  def values(self):
    return self._values

#***************************************************************
class NumericHyperparam(BaseHyperparam):
  """"""
  
  #=============================================================
  def __init__(self, section, option, bounds):
    """"""
    
    super(NumericHyperparam, self).__init__(section, option, bounds)
    self._process_bounds()
    self._lower = min(self.bounds)
    self._upper = max(self.bounds)
    return
  
  def _process_bounds(self):
    raise NotImplementedError()

  #=============================================================
  def normalize(self, value):
    """"""
    
    if self.fixed or (value is None):
      return np.nan
    else:
      return (value - self.lower.astype(np.int64)) / (self.upper.astype(np.int64) - self.lower.astype(np.int64))
    
  def denormalize(self, value):
    """"""
    
    if self.fixed or (value is None):
      return np.nan
    else:
      return value * (self.upper.astype(np.int64) - self.lower.astype(np.int64)) + self.lower.astype(np.int64)
  
  #=============================================================
  @property
  def upper(self):
    return self._upper
  @property
  def lower(self):
    return self._lower
