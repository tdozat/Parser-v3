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
from hpo.base_hpo import BaseHPO

def pretty_plot(ax):
  ax.grid(linestyle='--', axis='y')
  
#***************************************************************
class PPPHPO(BaseHPO):
  """"""
  
  #=============================================================
  def rand(self, dims=None):
    """"""
    
    if dims is not None:
      dims = set(dims)
    
    hps = [hp for i, hp in enumerate(self) if (not hp.fixed) and ((dims is None) or (i in dims))]
    n = len(self.scores)
    
    accepted = False
    attempts = 0
    best = None
    best_p = 0
    while not accepted:
      rand_dict = super(PPPHPO, self).rand()
      volume = 1
      mass = set(six.moves.range(len(hps)))
      d = sum([rand_dict[hp.section][hp.option] != np.nan for hp in hps])
      for hp in hps:
        if not accepted:
          choice = rand_dict[hp.section][hp.option]
          if choice != np.nan:
            hp_volume, hp_mass = hp.PPP_volume(choice, d)
            volume *= hp_volume
            mass.intersection_update(hp_mass)
            if len(mass) == 0:
              accepted = True
            for section, option in hp.copies:
              rand_dict[section][option] = choice
        else:
          break
      if not accepted:
        lamda = (n+1)*volume
        p = sps.poisson.sf(len(mass), lamda)
        accepted = sps.bernoulli.rvs(p)
        if not accepted:
          attempts += 1
          if p > best_p:
            best = rand_dict
            best_p = p
          if attempts == 15:
            accepted = True
            rand_dict = best
    
    return rand_dict
  
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
  def eval_func(save_dir):
    return evals.evaluate_tokens('data/CoNLL18/UD_English-EWT/en_ewt-ud-dev.conllu', os.path.join(save_dir, 'parsed/data/CoNLL18/UD_English-EWT/en_ewt-ud-dev.conllu'))
  #-------------------------------------------------------------
  hpo = PPPHPO('hpo/config/test.csv', 'saves/English', eval_func)
  rand_dict = next(hpo)
  n = 0
  for section in rand_dict:
    for option in rand_dict[section]:
      print('{}\t{}\t{}'.format(section, option, rand_dict[section][option]))
      n += 1
