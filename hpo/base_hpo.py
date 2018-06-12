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
import pandas as pd
import matplotlib.pyplot as plt

from hpo.hps import *

def pretty_plot(ax):
  ax.grid(linestyle='--', axis='y')
  
#***************************************************************
class BaseHPO(list):
  """"""
  
  #=============================================================
  def __init__(self, rand_csv, meta_save_dir, eval_func):
    """"""
    
    super(BaseHPO, self).__init__()
    rand_df = pd.read_csv(rand_csv)
    name2hp = {}
    
    # Get all non-copy hyperparams
    for row in rand_df.itertuples():
      if not row.bounds.startswith('%'):
        if row.dtype == 'str':
          hp = StringHyperparam(row.section, row.option, row.bounds)
        elif row.dtype == 'float':
          hp = FloatHyperparam(row.section, row.option, row.bounds)
        elif row.dtype == 'log':
          hp = LogHyperparam(row.section, row.option, row.bounds)
        elif row.dtype == 'int':
          hp = IntHyperparam(row.section, row.option, row.bounds)
        elif row.dtype == 'bool':
          hp = BoolHyperparam(row.section, row.option, row.bounds)
        self.append(hp)
        name2hp[hp.name] = hp
    
    # Get all copy hyperparams
    for row in rand_df.itertuples():
      if row.bounds.startswith('%'):
        copyrow = row
        while copyrow.bounds.startswith('%'):
          assert row.dtype == copyrow.dtype
          match = re.match('%\((\w+):(\w+)\)', copyrow.bounds)
          section = match.group(1)
          option = match.group(2)
          copyrow = rand_df.loc[(rand_df.section==section) & (rand_df.option==option)].iloc[0]
        name2hp[section + ':' + option].add_copy(row.section, row.option)
    
    indices = []
    scores = []
    for base_dir in os.listdir(meta_save_dir):
      save_dir = os.path.join(meta_save_dir, base_dir)
      if os.path.exists(os.path.join(save_dir, 'SUCCESS')):
        try:
          score = eval_func(save_dir)
        except IndexError:
          score = np.nan
        if np.isfinite(score):
          scores.append(score)
          indices.append(int(base_dir))
          config = SafeConfigParser()
          config.read(os.path.join(save_dir, 'config.cfg'))
          for hyperparam in self:
            hyperparam.add_config_value(config)
    scores = np.array(scores)
    median = np.median(scores)
    scores -= median
    scores /= np.median(np.abs(scores))
    to_remove = scores < -3
    scores = scores[(1-to_remove).astype(bool)]
    to_remove = np.where(to_remove)[0]
    for index in to_remove[::-1]:
      indices.pop(index)
    order = np.argsort(indices)
    self._scores = scores[order]
    for hp in self:
      hp.sort(order)
    return
  
  #=============================================================
  def rand(self):
    """"""
    
    rand_dict = DefaultDict(dict)
    for hp in self:
      rand_dict[hp.section][hp.option] = hp.rand()
    return rand_dict
  
  #=============================================================
  def improve_rank(self, dims=None):
    """"""
    
    hps = [hp for i, hp in enumerate(self) if (not hp.fixed) and ((dims is None) or (i in dims))]
    d = len(hps)
    discretes = [hp for hp in hps if isinstance(hp, (BoolHyperparam, StringHyperparam))]
    
    discrete_values = {}
    for discrete in discretes:
      for i, bound in enumerate(discrete.bounds):
        discrete_values[(discrete.name, bound)] = np.equal(discrete.values, i).astype(int)
    
    discrete_list = []
    for i, discrete1 in enumerate(discretes):
      for bound1 in discrete1.bounds:
        values1 = discrete_values[(discrete1.name, bound1)]
        for j, discrete2 in enumerate(discretes):
          if i != j:
            for bound2 in discrete2.bounds:
              values2 = discrete_values[(discrete2.name, bound2)]
              discrete_list.append((np.dot(values1, values2), discrete1.name, bound1, discrete2.name, bound2))
          elif i == j:
            discrete_list.append((np.dot(values1, values1), discrete1.name, bound1, discrete1.name, bound1))
        discrete_list.sort()
    
    rand_dict = DefaultDict(dict)
    hps_to_remove = set()
    for count, name1, bound1, name2, bound2 in discrete_list:
      section1, option1 = name1.split(':')
      section2, option2 = name2.split(':')
      if not (section1 in rand_dict and option1 in rand_dict[section1]) or\
         (section1 in rand_dict and option1 in rand_dict[section1] and bound1 == rand_dict[section1][option1]):
        if not (section2 in rand_dict and option2 in rand_dict[section2]) or\
           (section2 in rand_dict and option2 in rand_dict[section2] and bound2 == rand_dict[section2][option2]):
          if count <= d:
            rand_dict[section1][option1] = bound1
            rand_dict[section2][option2] = bound2
            hps_to_remove.add((section1, option1))
            hps_to_remove.add((section2, option2))
    if dims is not None or hps_to_remove:
      dims = [i for i, hp in enumerate(self) if ((i in dims) and ((hp.section, hp.option) not in hps_to_remove))]
    return rand_dict, np.array(dims)
  
  #=============================================================
  def clean_dict(self, rand_dict):
    """"""
    
    section = 'Network'
    if section in rand_dict:
      trigger = 'highway'
      if trigger in rand_dict[section] and not rand_dict[section][trigger]:
        if np.random.binomial(1, .5):
          rand_dict[section]['highway_func'] = None
        else:
          rand_dict[section][trigger] = True
      trigger = 'bidirectional'
      if trigger in rand_dict[section] and not rand_dict[section][trigger]:
        if np.random.binomial(1, .5):
          rand_dict[section]['bilin'] = None
        else:
          rand_dict[section][trigger] = True
      trigger = 'switch_optimizers'
      if trigger in rand_dict[section] and not rand_dict[section][trigger]:
        if np.random.binomial(1, .5):
          rand_dict['AMSGradOptimizer']['learning_rate'] = None
          rand_dict['AMSGradOptimizer']['decay_rate'] = None
          rand_dict['AMSGradOptimizer']['clip'] = None
          rand_dict['AMSGradOptimizer']['mu'] = None
          rand_dict['AMSGradOptimizer']['nu'] = None
          rand_dict['AMSGradOptimizer']['epsilon'] = None
          rand_dict['AMSGradOptimizer']['gamma'] = None
        else:
          rand_dict[section][trigger] = True
    
    section = 'TokenVocab'
    if section in rand_dict:
      if 'n_layers' in rand_dict[section] and rand_dict[section]['n_layers'] <= 0:
        rand_dict[section]['n_layers'] = 0
        rand_dict[section]['hidden_func'] = None
        rand_dict[section]['hidden_size'] = None
        
    for hp in self:
      value = rand_dict[hp.section][hp.option]
      for section, option in hp.copies:
        
        rand_dict[section][option] = value
    return rand_dict
  
  #=============================================================
  def plots(self):
    """"""
    
    scores = -self.scores + np.max(self.scores)
    k = sps.gamma.fit(scores, 2.5)
    x = np.linspace(np.min(scores), np.max(scores))
    dist = sps.gamma.pdf(x, *k)
    x = -x+np.max(self.scores)
    
    fig, ax = plt.subplots()
    ax.set_title('HPO', y=1.08)
    ax.set_ylabel('Normalized LAS')
    ax.set_xlabel('Iteration')
    ax.plot(self.scores)
    axt = ax.twiny()
    axt.plot(dist, x)
    axt.fill_between([np.min(dist)] + list(dist) + [np.min(dist)], [np.max(x)] + list(x) + [np.min(x)], alpha=.25)
    pretty_plot(ax)
    fig.tight_layout()
    plt.show()
    for hp in self:
      if not hp.fixed:
        hp.plot(self.scores)
    return
  
  #=============================================================
  def __next__(self):
    """"""
    
    rand_dict = self.rand()
    return self.clean_dict(rand_dict)
  
  #=============================================================
  @property
  def scores(self):
    return self._scores

#***************************************************************
if __name__ == '__main__':
  """"""
  
  from hpo import evals
  #-------------------------------------------------------------
  base_dir = 'hpo/English'
  base_dir = 'hpo/French'
  def eval_func(save_dir):
    base = 'data/CoNLL18/UD_English-EWT/en_ewt-ud-dev.conllu'
    base = 'data/CoNLL18/UD_French-Spoken/fr_spoken-ud-dev.conllu'
    return evals.evaluate_tokens(base, os.path.join(save_dir, 'parsed', base), force=False)
  #-------------------------------------------------------------
  hpo = BaseHPO('hpo/config/test.csv', base_dir, eval_func)
  rand_dict = next(hpo)
  n = 0
  for section in rand_dict:
    for option in rand_dict[section]:
      print('{}\t{}\t{}'.format(section, option, rand_dict[section][option]))
      n += 1
  hpo.plots()
