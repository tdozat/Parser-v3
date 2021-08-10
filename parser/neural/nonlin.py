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

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow import identity, tanh, asinh
from tensorflow.python.ops.nn import relu, elu

from parser.neural import nn

#===============================================================
# Base functions:
# * relu
# * tanh
# * asinh
# Relu combinations:
# * thlu (tanh < 0, relu > 0)
# * rethu (relu < 0, tanh > 0)
# * ashlu (arcsinh < 0, relu > 0)
# * reashu (relu < 0, arcsinh > 0)
def sigmoid(t):
  return tf.nn.sigmoid(2*t)

def rethu(t):
  return nn.where(t < 0, 0, tanh(t))

def reashu(t):
  return nn.where(t < 0, 0, asinh(t))

def thlu(t):
  return nn.where(t < 0, tanh(t), t)

def ashlu(t):
  return nn.where(t < 0, asinh(t), t)

def hard_tanh(t):
  return tf.clip_by_value(t, -1, 1)

def softplus(t):
  return .5*tf.log(tf.exp(2*t) + 1)

def centered_softplus(t):
  return softplus(t) - .5*tf.log(2.)

def glu(t):
  t, g = tf.split(t, 2, -1)
  return t * sigmoid(g)

def leaky_relu(t):
  return nn.where(t < 0, .1*t, t)

def log_relu(t):
  return tf.log(1+tf.nn.relu(t))
