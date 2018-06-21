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
import tensorflow as tf

from . import nn
from . import nonlin

#***************************************************************
def LSTM_func(inputs, cell, recur_func=tf.tanh, cifg=False):
  """"""
  
  n_splits = 4 - cifg
  splits = tf.split(inputs, n_splits, axis=1)
  activation = splits.pop(0)
  input_activation = splits.pop(0)
  
  if cifg:
    input_gate = tf.nn.sigmoid(2*input_activation)
    forget_gate = 1-input_gate
  else:
    forget_activation = splits.pop(0)
    activation = tf.tanh(activation)
    input_gate = tf.nn.sigmoid(2*input_activation)
    forget_gate = tf.nn.sigmoid(2*forget_activation)
  
  output_activation = splits.pop(0)
  output_gate = tf.nn.sigmoid(2*output_activation)
  
  cell = input_gate*activation + forget_gate*cell
  hidden = output_gate*recur_func(cell)
  return hidden, cell

#===============================================================
def RNN(layer, recur_size, conv_width=0, recur_func=nonlin.relu, conv_keep_prob=1., recur_keep_prob=1., recur_include_prob=1., **kwargs):
  """"""
  
  batch_size, bucket_size, input_size = nn.get_sizes(layer)
  conv_size = 1+(2*conv_width)
  
  weights = tf.get_variable('Weights', shape=[conv_size, input_size, recur_size], initializer=tf.orthogonal_initializer)
  biases = tf.get_variable('Biases', shape=recur_size, initializer=tf.zeros_initializer)
  if conv_keep_prob < 1:
    layer = nn.dropout(layer, conv_keep_prob, noise_shape=[batch_size, 1, input_size])
  layer = tf.nn.convolution(layer, weights, 'SAME') + biases
  
  with tf.variable_scope('Loop'):
    # Set up the variables
    i = tf.constant(0, dtype=tf.int32)
    input_sequence = tf.TensorArray(tf.float32, size=bucket_size)
    input_sequence = input_sequence.unstack(tf.transpose(layer, [1,0,2]))
    hidden_sequence = tf.TensorArray(tf.float32, size=bucket_size)
    weights = tf.get_variable('Weights', shape=[recur_size, recur_size], initializer=tf.zeros_initializer)
    initial_hidden = tf.get_variable('Initial_hidden', shape=[1, recur_size], initializer=tf.zeros_initializer)
    initial_hidden = nn.tile(initial_hidden, [batch_size, 1])
    null_hidden = tf.zeros_like(initial_hidden)
    mask = nn.drop_mask(tf.stack([batch_size, recur_size]), recur_keep_prob)
    
    # Set up the loop
    #-------------------------------------------------------------
    def cond(i, *args):
      return i < bucket_size
    #-------------------------------------------------------------
    def body(i, last_hidden, last_hidden_sequence):
      current_partial_input = input_sequence.read(i)
      if recur_keep_prob < 1:
        last_hidden *= mask
      current_hidden = current_partial_input + tf.matmul(last_hidden, weights)
      current_hidden = recur_func(current_hidden)
      if recur_include_prob < 1:
        zone_mask = nn.binary_mask([batch_size, recur_size], recur_include_prob)
        current_hidden = zone_mask*current_hidden + (1-zone_mask)*last_hidden
      current_hidden = tf.where(i < seq_lengths, current_hidden, null_hidden)
      current_hidden_sequence = last_hidden_sequence.write(i, tf.where(i < seq_lengths, current_hidden, null_hidden))
      return (i+1, current_hidden, current_hidden_sequence)
    #-------------------------------------------------------------
    loop_vars = (i, initial_hidden, hidden_sequence)
    #-------------------------------------------------------------
    
    # Run the loop
    _, _, final_hidden_sequence = tf.while_loop(
      cond=cond,
      body=body,
      loop_vars=loop_vars)
    
    # Get the outputs
    layer = tf.transpose(final_hidden_sequence.stack(), [1,0,2])
  return layer, layer

#===============================================================
def LSTM(layer, recur_size, seq_lengths, conv_width=0, recur_func=nonlin.tanh, conv_keep_prob=1., recur_keep_prob=1., recur_include_prob=1., cifg=False, highway=False, highway_func=tf.identity):
  """"""
  
  #max_length = tf.reduce_max(seq_lengths)
  batch_size, bucket_size, input_size = nn.get_sizes(layer)
  gate_size = (3-cifg)*recur_size
  highway_size = highway*2*recur_size
  gated_size = gate_size + recur_size
  gated_highway_size = gated_size + highway_size
  conv_size = 1+(2*conv_width)
  
  weights = tf.get_variable('Weights', shape=[conv_size, input_size, recur_size], initializer=tf.orthogonal_initializer)
  gate_weights = tf.get_variable('Gate_Weights', shape=[conv_size, input_size, gate_size], initializer=tf.orthogonal_initializer)
  highway_weights = tf.get_variable('Highway_Weights', shape=[conv_size, input_size, highway_size], initializer=tf.orthogonal_initializer)
  weights = tf.concat([weights, gate_weights, highway_weights], axis=2)
  biases = tf.get_variable('Biases', shape=gated_highway_size, initializer=tf.zeros_initializer)
  if conv_keep_prob < 1:
    layer = nn.dropout(layer, conv_keep_prob, noise_shape=[batch_size, 1, input_size])
  layer = tf.nn.convolution(layer, weights, 'SAME') + biases
  if highway:
    layer, highway_layer = tf.split(layer, [gated_size, highway_size], axis=2)
  
  with tf.variable_scope('Loop'):
    # Set up the variables
    i = tf.constant(0, dtype=tf.int32)
    input_sequence = tf.TensorArray(tf.float32, size=bucket_size)
    input_sequence = input_sequence.unstack(tf.transpose(layer, [1,0,2]))
    state_sequence = tf.TensorArray(tf.float32, size=bucket_size)
    weights = tf.get_variable('Weights', shape=[recur_size, recur_size], initializer=tf.orthogonal_initializer) # TODO try zeros_initializer
    gate_weights = tf.get_variable('Gate_Weights', shape=[recur_size, gate_size], initializer=tf.orthogonal_initializer)
    weights = tf.concat([weights, gate_weights], axis=1)
    initial_state = tf.get_variable('Initial_state', shape=[1, 2*recur_size], initializer=tf.zeros_initializer)
    initial_state = nn.tile(initial_state, [batch_size, 1])
    null_state = tf.zeros_like(initial_state)
    mask = nn.drop_mask([batch_size, recur_size], recur_keep_prob)
    
    # Set up the loop
    #-------------------------------------------------------------
    def cond(i, *args):
      return i < bucket_size 
    #-------------------------------------------------------------
    def body(i, last_state, last_state_sequence):
      last_hidden, last_cell = tf.split(last_state, 2, axis=1)
      current_partial_input = input_sequence.read(i)
      if recur_keep_prob < 1:
        last_hidden *= mask
      current_input = current_partial_input + tf.matmul(last_hidden, weights)
      current_hidden, current_cell = LSTM_func(current_input, last_cell, recur_func=recur_func, cifg=cifg)
      if recur_include_prob < 1:
        zone_mask = nn.binary_mask([batch_size, recur_size], recur_include_prob)
        current_hidden = zone_mask*current_hidden + (1-zone_mask)*last_hidden
        current_cell = zone_mask*current_cell + (1-zone_mask)*last_cell
      current_state = tf.concat([current_hidden, current_cell], 1)
      current_state_sequence = last_state_sequence.write(i, current_state)
      #current_state = tf.where(i < seq_lengths, current_state, null_state)
      #current_state_sequence = last_state_sequence.write(i, tf.where(i < seq_lengths, current_state, null_state))
      return (i+1, current_state, current_state_sequence)
    #-------------------------------------------------------------
    loop_vars = (i, initial_state, state_sequence)
    #-------------------------------------------------------------
    
    # Run the loop
    _, _, final_state_sequence = tf.while_loop(
      cond=cond,
      body=body,
      loop_vars=loop_vars)
    
    # Get the outputs
    layer = tf.transpose(final_state_sequence.stack(), [1,0,2])
    hidden, cell = tf.split(layer, 2, axis=2)
  
  if highway:
    activation, gate = tf.split(highway_layer, 2, axis=2)
    activation = highway_func(activation)
    gate = tf.nn.sigmoid(2*gate)
    activation = gate*activation
    hidden += activation
  return hidden, cell
  
#===============================================================
def directed_RNN(layer, recur_size, seq_lengths, bidirectional=True, recur_cell=LSTM, **kwargs):
  """"""
  
  bilin = kwargs.pop('bilin', False)
  if bidirectional:
    return BiRNN(layer, recur_size, seq_lengths, recur_cell=recur_cell, bilin=bilin, **kwargs)
  else:
    return UniRNN(layer, recur_size, seq_lengths, recur_cell=recur_cell, **kwargs)

#===============================================================
def UniRNN(layer, recur_size, seq_lengths, recur_cell=LSTM, **kwargs):
  """"""
  
  locations = tf.expand_dims(tf.one_hot(seq_lengths-1, tf.shape(layer)[1]), -1)
  with tf.variable_scope('RNN'):
    hidden, cell = recur_cell(layer, recur_size, seq_lengths, **kwargs)
  layer = hidden
  if recur_cell == RNN:
    final_states = tf.squeeze(tf.matmul(hidden, locations, transpose_a=True), -1)
  else:
    final_hidden = tf.squeeze(tf.matmul(hidden, locations, transpose_a=True), -1)
    final_cell = tf.squeeze(tf.matmul(cell, locations, transpose_a=True), -1)
    final_states = tf.concat([final_hidden, final_cell], 1)
  return layer, final_states

#===============================================================
def BiRNN(layer, recur_size, seq_lengths, recur_cell=LSTM, bilin=False, **kwargs):
  """"""
  
  locations = tf.expand_dims(tf.one_hot(seq_lengths-1, tf.shape(layer)[1]), -1)
  with tf.variable_scope('RNN_FW'):
    fw_hidden, fw_cell = recur_cell(layer, recur_size, seq_lengths, **kwargs)
  rev_layer = tf.reverse_sequence(layer, seq_lengths, batch_axis=0, seq_axis=1)
  with tf.variable_scope('RNN_BW'):
    bw_hidden, bw_cell = recur_cell(rev_layer, recur_size, seq_lengths, **kwargs)
  rev_bw_hidden = tf.reverse_sequence(bw_hidden, seq_lengths, batch_axis=0, seq_axis=1)
  rev_bw_cell = tf.reverse_sequence(bw_cell, seq_lengths, batch_axis=0, seq_axis=1)
  if bilin:
    layer = tf.concat([fw_hidden*rev_bw_hidden, fw_hidden, rev_bw_hidden], 2)
  else:
    layer = tf.concat([fw_hidden, rev_bw_hidden], 2)
  if recur_cell == RNN:
    final_states = tf.squeeze(tf.matmul(hidden, locations, transpose_a=True), -1)
  else:
    final_fw_hidden = tf.squeeze(tf.matmul(fw_hidden, locations, transpose_a=True), -1)
    final_fw_cell = tf.squeeze(tf.matmul(fw_cell, locations, transpose_a=True), -1)
    final_rev_bw_hidden = tf.squeeze(tf.matmul(rev_bw_hidden, locations, transpose_a=True), -1)
    final_rev_bw_cell = tf.squeeze(tf.matmul(rev_bw_cell, locations, transpose_a=True), -1)
    final_states = tf.concat([final_fw_hidden, final_rev_bw_hidden, final_fw_cell, final_rev_bw_cell], 1)
  return layer, final_states
