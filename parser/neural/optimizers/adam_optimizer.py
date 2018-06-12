#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import tensorflow as tf

from parser.neural.optimizers.optimizer import Optimizer

#***************************************************************
class AdamOptimizer(Optimizer):
  """"""
  
  #=============================================================
  def dense_update(self, gradient, variable):
    """"""
    
    updates = []
    
    if self.mu > 0:
      mean_update = self.dense_moving_average(
        variable, gradient,
        name='Mean',
        decay=self.mu)
      mean, _ = mean_update
      mean = (1-self.gamma) * mean + self.gamma * gradient
      updates.extend(mean_update)
    else:
      mean = gradient
    
    if self.nu > 0:
      zero_deviation_update = self.dense_moving_average(
        variable, gradient**2,
        name='ZeroDeviation',
        decay=self.nu)
      zero_deviation, _ = zero_deviation_update
      zero_deviation = tf.sqrt(zero_deviation + self.epsilon)
      updates.extend(zero_deviation_update)
    else:
      zero_deviation = 1
    
    variable_step = self.annealed_learning_rate * mean / zero_deviation
    variable_step = tf.where(tf.is_finite(variable_step),
                               variable_step,
                               tf.zeros_like(variable_step))
    return variable_step, updates
  
  #=============================================================
  def sparse_update(self, gradient, variable):
    """"""
    
    updates = []
    
    unique_indices, sorting_indices = tf.unique(gradient.indices)
    indexed_gradient = tf.unsorted_segment_sum(gradient.values, sorting_indices, tf.size(unique_indices))
    if self.mu > 0:
      mean_update = self.sparse_moving_average(
        variable, unique_indices, indexed_gradient,
        name='Mean',
        decay=self.mu)
      mean, _ = mean_update
      indexed_mean = tf.gather(mean, unique_indices)
      indexed_mean = (1-self.gamma) * indexed_mean + self.gamma * indexed_gradient
      updates.extend(mean_update)
    else:
      indexed_mean = indexed_gradient
    
    if self.nu > 0:
      zero_deviation_update = self.sparse_moving_average(
        variable, unique_indices, indexed_gradient**2,
        name='ZeroDeviation',
        decay=self.nu)
      zero_deviation, _ = zero_deviation_update
      indexed_zero_deviation = tf.gather(zero_deviation, unique_indices)
      indexed_zero_deviation = tf.sqrt(indexed_zero_deviation + self.epsilon)
      updates.extend(zero_deviation_update)
    else:
      indexed_zero_deviation = 1
    
    indexed_variable_step = self.annealed_learning_rate * indexed_mean / indexed_zero_deviation
    indexed_variable_step = tf.where(tf.is_finite(indexed_variable_step),
                                      indexed_variable_step,
                                      tf.zeros_like(indexed_variable_step))
    return indexed_variable_step, unique_indices, updates
  
