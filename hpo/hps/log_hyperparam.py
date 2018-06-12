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
  
from hpo.hps.float_hyperparam import FloatHyperparam

#***************************************************************
class LogHyperparam(FloatHyperparam):
  """"""
  
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
    
    x = np.log10(np.array([self.denormalize(value) for value in self.values]))
    minx = np.min(x)
    maxx = np.max(x)
    _range = np.linspace(minx, maxx)
    
    mean = weights.dot(x)
    centered = x-mean
    var = centered.dot(np.diag(weights)).dot(centered) / (1-weights.dot(weights))
    dist = sps.norm.pdf(_range, mean, np.sqrt(var))
    
    x = x[:,None]
    X = np.concatenate([np.ones_like(x), x, x**2], axis=1).astype(float)
    theta = la.inv(X.T.dot(X)+.05*np.eye(3)).dot(X.T).dot(scores)
    b, w1, w2 = theta
    minx = np.min(x)
    maxx = np.max(x)
    lin = np.linspace(minx, maxx)
    curve = b + w1*lin + w2*lin**2
    optimum = -.5*w1/w2
    
    fig, ax = plt.subplots()
    ax.set_title(self.section)
    ax.set_ylabel('Normalized LAS')
    ax.set_xlabel(self.option)
    ax.scatter(x, scores, alpha=.5, edgecolor='k')
    ax.plot(lin, curve, color='c' if w2 < 0 else 'r')
    if optimum < maxx and optimum > minx:
      ax.axvline(optimum, ls='--')
    axt = ax.twinx()
    axt.plot(_range, dist)
    axt.fill_between(_range, dist, alpha=.25)
    pretty_plot(ax)
    plt.show()
    return
  
  def normalize(self, value):
    return super(LogHyperparam, self).normalize(np.log10(value))
  
  def denormalize(self, value):
    return 10**(super(LogHyperparam, self).denormalize(value))
  
  def _rand(self):
    return 10**np.random.uniform(self.lower, self.upper)
  
