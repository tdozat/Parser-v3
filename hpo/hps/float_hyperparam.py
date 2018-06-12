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
  
from hpo.hps.base_hyperparam import NumericHyperparam

#***************************************************************
class FloatHyperparam(NumericHyperparam):
  """"""
  
  #=============================================================
  def PPP_volume(self, choice, d):
    """"""
    
    n = len(np.isfinite(self.values))
    norm_choice = self.normalize(choice)
    lower_bound = norm_choice - .5/np.power(n+1, 1/d)
    upper_bound = norm_choice + .5/np.power(n+1, 1/d)
    lower = max(0, lower_bound - max(0, upper_bound-1))
    upper = min(1, upper_bound - min(0, lower_bound))
    length = upper - lower
    cluster = set(np.where(np.greater_equal(self.values, lower) *
                           np.less_equal(self.values, upper))[0])
    return length, cluster
  
  #=============================================================
  def _process_bounds(self):
    self._bounds = [float(bound) for bound in self.bounds]
    return
  
  def get_config_value(self, config):
    return config.getfloat(self.section, self.option)
  
  def _rand(self):
    return np.random.uniform(self.lower, self.upper)
