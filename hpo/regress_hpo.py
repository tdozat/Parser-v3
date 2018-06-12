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

import os
from collections import defaultdict as DefaultDict

import numpy as np
import numpy.linalg as la
import scipy.stats as sps

from hpo.hps import *
from hpo.ppp_hpo import PPPHPO

#***************************************************************
class RegressHPO(PPPHPO):
  """"""
  
  #=============================================================
  def __init__(self, *args, k=2.5, **kwargs):
    """"""
    
    super(RegressHPO, self).__init__(*args, **kwargs)
    self._k = k
    return
  
  #=============================================================
  def rand(self, dims=None):
    """"""
    
    if dims is not None:
      dims = set(dims)
    
    hps = [hp for i, hp in enumerate(self) if (not hp.fixed) and ((dims is None) or (i in dims))]
    
    mat = [np.ones([len(self.scores),1])] + [hp.as_matrix() for hp in hps]
    mat = np.concatenate(mat, axis=1)
    d = mat.shape[1]-1
    
    interactmat = []
    for i, vec1 in enumerate(mat.T):
      for j, vec2 in enumerate(mat.T):
        if i <= j:
          interactmat.append((vec1*vec2)[:,None])
    
    X = np.concatenate(interactmat, axis=1)
    n, d2 = X.shape
    I = np.eye(d2)
    I[0,0] = 0
    XTXinv = la.inv(X.T.dot(X) + .05*I)
    # TODO maybe: L1 regularization on the interactions
    mean = XTXinv.dot(X.T).dot(self.scores)
    H = X.dot(XTXinv).dot(X.T)
    epsilon_hat = self.scores - H.dot(self.scores)
    dof = np.trace(np.eye(n) - H)
    s_squared = epsilon_hat.dot(epsilon_hat) / dof
    cov = s_squared * XTXinv
    eigenvals, eigenvecs = la.eig(cov)
    eigenvals = np.diag(np.abs(eigenvals))
    eigenvecs = np.real(eigenvecs)
    cov = eigenvecs.dot(eigenvals).dot(eigenvecs.T)
    cov += .05**2*np.eye(len(cov))
    if la.matrix_rank(cov) < len(cov):
      print('WARNING: indefinite covariance matrix')
      return {}
    
    rand_dict = DefaultDict(dict)
    vals = np.random.multivariate_normal(mean, cov)
    bias = vals[0]
    lins = vals[1:d+1]
    bilins = np.zeros([d,d])
    bilins[np.tril_indices(d)] = vals[d+1:]
    bilins = .5*bilins + .5*bilins.T
    eigenvals, eigenvecs = la.eig(bilins)
    eigenvals = -np.diag(np.abs(eigenvals))
    eigenvecs = np.real(eigenvecs)
    bilins = eigenvecs.dot(eigenvals).dot(eigenvecs.T)
    if la.matrix_rank(bilins) < len(bilins):
      print('WARNING: indefinite interaction matrix')
      return {}
    
    rand_dict = DefaultDict(dict)
    vals = np.clip(.5*la.inv(bilins).dot(lins), 0, 1)
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
    
    a = .5
    b = 1.5
    c = (2*self.k - len(self.scores)) / self.k
    q = int( (-b + np.sqrt(b**2 - 4*a*c)) / (2*a) )
    maximize = np.random.permutation(np.arange(len(self)))[:q]
    
    rand_dict = super(RegressHPO, self).rand()
    rank_dict, maximize = self.improve_rank(maximize)
    for k, v in six.iteritems(rank_dict):
      rand_dict[k].update(v)
    max_dict = self.rand(maximize)
    for k, v in six.iteritems(max_dict):
      rand_dict[k].update(v)
    return self.clean_dict(rand_dict)
  
  #=============================================================
  @property
  def k(self):
    return self._k

#***************************************************************
if __name__ == '__main__':
  """"""
  
  from hpo import evals
  
  #-------------------------------------------------------------
  def eval_func(save_dir):
    return evals.evaluate_tokens('data/CoNLL18/UD_English-EWT/en_ewt-ud-dev.conllu', os.path.join(save_dir, 'parsed/data/CoNLL18/UD_English-EWT/en_ewt-ud-dev.conllu'))
  #-------------------------------------------------------------
  hpo = RegressHPO('hpo/config/test.csv', 'saves/English', eval_func)
  rand_dict = next(hpo)
  n = 0
  for section in rand_dict:
    for option in rand_dict[section]:
      print('{}\t{}\t{}'.format(section, option, rand_dict[section][option]))
      n += 1
