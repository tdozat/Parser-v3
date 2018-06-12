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
import six

import re
import tensorflow as tf

#***************************************************************
class Optimizer(object):
  """"""
  
  #=============================================================
  def __init__(self, config=None):
    """"""
    
    self._accumulators = {}
    self._global_step = tf.Variable(0., trainable=False, name='global_step')
    self._config = config
    return
  
  #=============================================================
  @classmethod
  def from_optimizer(cls, optimizer):
    """"""
    
    new_optimizer = cls(config=optimizer._config)
    new_optimizer._accumulators = optimizer._accumulators
    new_optimizer._global_step = optimizer._global_step
    return new_optimizer
  
  #=============================================================
  def minimize(self, loss, variables=None):
    """"""
    
    variables = variables or tf.trainable_variables()
    gradients = tf.gradients(loss, variables,
                             colocate_gradients_with_ops=True,
                             gate_gradients=True,
                             aggregation_method=2)
    gradients = {variable: gradient for variable, gradient in zip(variables, gradients) if gradient is not None}
    
    variable_steps = {}
    variable_indices = {}
    updates = [tf.assign_add(self.global_step, 1)]
    for variable, gradient in six.iteritems(gradients):
      if isinstance(gradient, tf.Tensor):
        step, update = self.dense_update(gradient, variable)
        variable_steps[variable] = step
        updates.extend(update)
      else:
        step, indices, update = self.sparse_update(gradient, variable)
        variable_steps[variable] = step
        variable_indices[variable] = indices
        updates.extend(update)
    
    variable_steps = self.clip_by_global_norm(variable_steps)
    
    for variable, step in six.iteritems(variable_steps):
      if variable in variable_indices:
        indices = variable_indices[variable]
        updates.append(tf.scatter_sub(variable, indices, step))
      else:
        updates.append(tf.assign_sub(variable, step))
    
    return tf.tuple(updates)[0]
  
  #=============================================================
  def dense_adam_update(self, gradient, variable):
    """"""
    
    raise NotImplementedError()
  
  #=============================================================
  def dense_moving_average(self, variable, accumulant, name='Accumulator', decay=.9):
    """"""
    
    accumulant = tf.clip_by_value(accumulant, -self.clip, self.clip)
    accumulator = self.get_accumulator(name, variable)
    iteration = self.get_accumulator('{}/iteration'.format(name), variable, shape=[])
    iteration = tf.assign_add(iteration, 1)
    
    if decay < 1:
      current_decay = decay * (1-decay**(iteration-1)) / (1-decay**iteration)
    else:
      current_decay = (iteration-1) / iteration
    
    accumulator = tf.assign(accumulator, current_decay*accumulator)
    accumulator = tf.assign_add(accumulator, (1-current_decay)*accumulant)
    return accumulator, iteration
  
  #=============================================================
  def sparse_update(self, gradient, variable):
    """"""
    
    raise NotImplementedError()
  
  #=============================================================
  def sparse_moving_average(self, variable, unique_indices, accumulant, name='Accumulator', decay=.9):
    """"""
    
    accumulant = tf.clip_by_value(accumulant, -self.clip, self.clip)
    first_dim = variable.get_shape().as_list()[0]
    accumulator = self.get_accumulator(name, variable)
    indexed_accumulator = tf.gather(accumulator, unique_indices)
    iteration = self.get_accumulator('{}/iteration'.format(name), variable, shape=[first_dim, 1])
    indexed_iteration = tf.gather(iteration, unique_indices)
    iteration = tf.scatter_add(iteration, unique_indices, tf.ones_like(indexed_iteration))
    indexed_iteration = tf.gather(iteration, unique_indices)
    
    if decay < 1:
      current_indexed_decay = decay * (1-decay**(indexed_iteration-1)) / (1-decay**indexed_iteration)
    else:
      current_indexed_decay = (indexed_iteration-1) / indexed_iteration
    
    accumulator = tf.scatter_update(accumulator, unique_indices, current_indexed_decay*indexed_accumulator)
    accumulator = tf.scatter_add(accumulator, unique_indices, (1-current_indexed_decay)*accumulant)
    return accumulator, iteration
  
  #=============================================================
  def clip_by_global_norm(self, variable_steps):
    """"""
    
    variable_step_list = list(variable_steps.values())
    variable_step_list, _ = tf.clip_by_global_norm(variable_step_list, self.clip)
    variable_steps = dict(zip(variable_steps.keys(), variable_step_list))
    return variable_steps
  
  #=============================================================
  def get_accumulator(self, name, original_variable, shape=None):
    """"""
    
    key = (original_variable, name)
    if key in self._accumulators:
      variable = self._accumulators[key]
    else:
      shape = shape if shape is not None else original_variable.get_shape().as_list()
      initializer = tf.zeros_initializer
      with tf.control_dependencies([original_variable]):
        with tf.variable_scope(original_variable.op.name, reuse=False):
          with tf.device(original_variable.device):
            variable = tf.get_variable(name, shape=shape,
                                       initializer=initializer,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
                                       trainable=False)
      self._accumulators[key] = variable
    return variable
  
  #=============================================================
  @property
  def learning_rate(self):
    return self._config.getfloat(self, 'learning_rate') 
  @property
  def decay_rate(self):
    return self._config.getfloat(self, 'decay_rate') 
  @property
  def annealed_learning_rate(self):
    return self.learning_rate * tf.exp(-self.decay_rate*self.global_step)
  @property
  def mu(self):
    return self._config.getfloat(self, 'mu')
  @property
  def nu(self):
    return self._config.getfloat(self, 'nu')
  @property
  def gamma(self):
    return self._config.getfloat(self, 'gamma')
  @property
  def clip(self):
    return self._config.getfloat(self, 'clip')
  @property
  def epsilon(self):
    return self._config.getfloat(self, 'epsilon')
  @property
  def global_step(self):
    return self._global_step
