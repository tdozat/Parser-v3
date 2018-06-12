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
class IntHyperparam(NumericHyperparam):
  """"""
  
  #=============================================================
  def PPP_volume(self, choice, d):
    """"""
    
    n = len(self.values)
    norm_choice = self.normalize(choice)
    lower_bound = int(np.round(norm_choice - .5/np.power(n+1, 1/d)))
    upper_bound = int(np.round(norm_choice + .5/np.power(n+1, 1/d)))
    lower = max(0, lower_bound - max(0, upper_bound-1))
    upper = min(1, upper_bound - min(0, lower_bound))
    length = upper - lower + 1
    cluster = set(np.where(np.greater_equal(self.values, lower) * 
                           np.less_equal(self.values, upper))[0])
    return length, cluster
  
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
    minx = np.min(x)-.5
    maxx = np.max(x)+.5
    _range = np.linspace(minx, maxx)
    
    mean = weights.dot(x)
    centered = x-mean
    var = centered.dot(np.diag(weights)).dot(centered) / (1-weights.dot(weights))
    dist = sps.norm.pdf(_range, mean, np.sqrt(var))
    
    fig, ax = plt.subplots()
    ax.set_title(self.section)
    ax.set_ylabel('Normalized LAS')
    ax.set_xlabel(self.option)
    
    d = len(np.unique(self.values))
    print(np.unique(self.values))
    x = x[:,None]
    if d < 5:
      violin = []
      for i in six.moves.range(d):
        violin.append(scores[np.where(np.equal(self.values, i))])
      ax.violinplot(violin, np.arange(d), showmeans=True)
      ax.set_xticks(np.arange(d))
    else:
      X = np.concatenate([np.ones_like(x), x, x**2], axis=1).astype(float)
      theta = la.inv(X.T.dot(X)+.05*np.eye(3)).dot(X.T).dot(scores)
      b, w1, w2 = theta
      curve = b + w1*_range + w2*_range**2
      optimum = -.5*w1/w2
      ax.plot(_range, curve, color='c' if w2 < 0 else 'r')
      if optimum < maxx and optimum > minx:
        ax.axvline(optimum, ls='--', color='c' if w2 < 0 else 'r')
    axt = ax.twinx()
    axt.plot(_range, dist)
    axt.fill_between(_range, dist, alpha=.25)
    
    ax.scatter(x, scores, alpha=.5, edgecolor='k')
    pretty_plot(ax)
    plt.show()
    return
  
  #=============================================================
  def denormalize(self, value):
    return int(np.round(super(IntHyperparam, self).denormalize(value)))
    
  def _process_bounds(self):
    self._bounds = [int(bound) for bound in self.bounds]
    return
    
  def get_config_value(self, config):
    return config.getint(self.section, self.option)

  def _rand(self):
    return np.random.randint(self.lower, self.upper+1)
  
