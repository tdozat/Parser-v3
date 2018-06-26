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

import re
import os
 
import pickle as pkl
import codecs
from collections import defaultdict as DefaultDict

try:
  from ConfigParser import SafeConfigParser
except ImportError:
  from configparser import SafeConfigParser

import numpy as np
import numpy.linalg as la
import scipy.stats as sps

from hpo.hps import *
from hpo.ppp_hpo import PPPHPO

def pretty_plot(ax):
  ax.grid(linestyle='--', axis='y')
  
#***************************************************************
class MVGHPO(PPPHPO):
  """"""
  
  #=============================================================
  def __init__(self, *args, lamda=.983, **kwargs):
    """"""
    
    super(MVGHPO, self).__init__(*args, **kwargs)
    self._lamda = lamda
    return
  
  #=============================================================
  def rand(self, dims=None):
    """"""
    
    if dims is not None:
      dims = set(dims)
    
    #-----------------------------------------------------------
    def compute_weights(scores):
      """ computes softmax(log(len(scores)) * scores) """
      
      scores = scores - np.max(scores)
      exp_scores = len(scores)**scores
      weights = exp_scores / exp_scores.sum()
      return weights
    #-----------------------------------------------------------
    
    finite_scores = np.isfinite(self.scores)
    hps = [hp for i, hp in enumerate(self) if (not hp.fixed) and ((dims is None) or (i in dims))]
    d = 0
    for hp in hps:
      if isinstance(hp, StringHyperparam):
        d += len(hp.bounds)-1
      else:
        d += 1
    
    # Compute the weighted means
    mean = np.zeros(d)
    i = 0
    for hp in hps:
      locs = np.where(np.isfinite(hp.values) * finite_scores)[0]
      if isinstance(hp, NumericHyperparam):
        all_values = hp.values[locs][None,:]
      else:
        all_values = np.zeros([len(hp.bounds)-1, len(locs)])
        for j in six.moves.range(len(hp.bounds)-1):
          all_values[j, np.where(hp.values[locs] == j)] = 1
      weights = compute_weights(self.scores[locs])
      for j, values in enumerate(all_values):
        mean[i+j] = np.dot(values, weights)
      i += j+1
    
    # Compute the weighted covariance
    cov = np.zeros([d, d])
    i1 = 0
    for hp1 in hps:
      i2 = 0
      for hp2 in hps:
        locs = np.where(np.isfinite(hp1.values) * np.isfinite(hp2.values) * finite_scores)[0]
        if isinstance(hp1, NumericHyperparam):
          all_values1 = hp1.values[locs][None,:]
        else:
          all_values1 = np.zeros([len(hp1.bounds)-1, len(locs)])
          for j in six.moves.range(len(hp1.bounds)-1):
            all_values1[j, np.where(hp1.values[locs] == j)] = 1
        if isinstance(hp2, NumericHyperparam):
          all_values2 = hp2.values[locs][None,:]
        else:
          all_values2 = np.zeros([len(hp2.bounds)-1, len(locs)])
          for j in six.moves.range(len(hp2.bounds)-1):
            all_values2[j, np.where(hp2.values[locs] == j)] = 1
        for j1, values1 in enumerate(all_values1):
          values1 = hp1.values[locs] - mean[i1+j1]
          bool_values1 = hp1.values[locs].astype(bool)
          for j2, values2 in enumerate(all_values2):
            if j2 <= j1:
              values2 = hp2.values[locs] - mean[i2+j2]
              bool_values2 = hp2.values[locs].astype(bool)
              weights = compute_weights(self.scores[locs])
              value = np.dot(weights*values1, values2) / (1 - np.dot(weights, weights))
              cov[i1+j1, i2+j2] = cov[i2+j2, i1+j1] = value
        i2 += j2+1
      i1 += j1+1
    cov += .05**2*np.eye(len(cov))
    eigenvals, eigenvecs = la.eig(cov)
    eigenvals = np.abs(eigenvals)
    cov = (np.abs(eigenvals[:,None]) * eigenvecs).dot(eigenvecs.T)
    if la.matrix_rank(cov) < len(cov):
      print('WARNING: indefinite covariance matrix')
      return {}
    
    rand_dict = DefaultDict(dict)
    vals = np.clip(np.random.multivariate_normal(mean, cov), 0, 1)
    i = 0
    for hp in hps:
      if isinstance(hp, NumericHyperparam):
        rand_dict[hp.section][hp.option] = hp.denormalize(vals[i])
        i += 1
      else:
        rand_dict[hp.section][hp.option] = hp.denormalize(vals[i:i+len(hp.bounds)-1])
        i += len(hp.bounds)-1
    return rand_dict
  
  
  #=============================================================
  def __next__(self):
    """"""
    
    p = min(1, self.lamda ** (len(self.scores)-(len(self)+2)))
    dims = np.random.binomial(1, p, len(self))
    maximize = np.arange(len(self))[(1-dims).astype(bool)]
    rand_dict = super(MVGHPO, self).rand()
    rank_dict, maximize = self.improve_rank(maximize)
    #if len(self.scores) < len(self)+2:
    if True:
      return self.clean_dict(rand_dict)

    for k, v in six.iteritems(rank_dict):
      rand_dict[k].update(v)
    max_dict = self.rand(maximize)
    print(max_dict)
    for k, v in six.iteritems(max_dict):
      rand_dict[k].update(v)
    return self.clean_dict(rand_dict)
  
  #=============================================================
  @property
  def lamda(self):
    return self._lamda

#***************************************************************
if __name__ == '__main__':
  """"""
  
  from hpo import evals
  
  #-------------------------------------------------------------
  def eval_func(save_dir):
    return evals.evaluate_tokens('data/CoNLL18/UD_English-EWT/en_ewt-ud-dev.conllu', os.path.join(save_dir, 'parsed/data/CoNLL18/UD_English-EWT/en_ewt-ud-dev.conllu'))
  #-------------------------------------------------------------
  hpo = MVGHPO('hpo/config/test.csv', 'saves/English', eval_func)
  rand_dict = next(hpo)
  n = 0
  for section in rand_dict:
    for option in rand_dict[section]:
      print('{}\t{}\t{}'.format(section, option, rand_dict[section][option]))
      n += 1
