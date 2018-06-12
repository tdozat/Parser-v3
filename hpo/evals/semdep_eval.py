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

from collections import Counter, namedtuple
import codecs
import sys
import os

#===============================================================
def compute_F1(gold_file, sys_file, labeled=False):
  """"""
  
  if not os.path.exists(sys_file):
    return None
  
  n_correct_edges = 0
  n_predicted_edges = 0
  n_actual_edges = 0
  
  n_correct_tokens = 0
  n_actual_tokens = 0
  n_tokens = 0
  
  n_correct_sequences = 0
  n_sequences = 0
  
  current_n_correct_edges = 0
  current_n_predicted_edges = 0
  current_n_actual_edges = 0
  
  current_n_correct_tokens = 0
  current_n_actual_tokens = 0
  current_n_tokens = 0
  
  current_n_correct_sequences = 1
  
  with codecs.open(gold_file, encoding='utf-8') as gf,\
       codecs.open(sys_file, encoding='utf-8') as sf:
    gold_line = gf.readline()
    while gold_line:
      while gold_line.startswith('#'):
        if current_n_tokens:
          n_correct_edges += current_n_correct_edges
          n_predicted_edges += current_n_predicted_edges
          n_actual_edges += current_n_actual_edges
          
          n_correct_tokens += current_n_correct_tokens
          n_actual_tokens += current_n_actual_tokens
          n_tokens += current_n_tokens
          
          n_correct_sequences += current_n_correct_sequences
          n_sequences += 1
          
        current_n_correct_edges = 0
        current_n_predicted_edges = 0
        current_n_actual_edges = 0
        
        current_n_correct_tokens = 0
        current_n_actual_tokens = 0
        current_n_tokens = 0
        
        current_n_correct_sequences = 1
        gold_line = gf.readline()
      
      if gold_line.rstrip() != '': 
        sys_line = sf.readline()
        while sys_line.rstrip() == '' or sys_line.split('\t')[0] == '0':
          sys_line = sf.readline()
        
        gold_line = gold_line.rstrip().split('\t')
        sys_line = sys_line.rstrip().split('\t')
        
        # Compute the gold edges
        gold_node = gold_line[8]
        if gold_node != '_':
          gold_node = gold_node.split('|')
          if labeled:
            gold_edges = set(tuple(gold_edge.split(':', 1)) for gold_edge in gold_node)
          else:
            gold_edges = set(gold_edge.split(':', 1)[0] for gold_edge in gold_node)
        else:
          gold_edges = set()
        
        # Compute the sys edges
        sys_node = sys_line[8]
        if sys_node != '_':
          sys_node = sys_node.split('|')
          if labeled:
            sys_edges = set(tuple(sys_edge.split(':', 1)) for sys_edge in sys_node)
          else:
            sys_edges = set(sys_edge.split(':', 1)[0] for sys_edge in sys_node)
        else:
          sys_edges = set()
        
        # Compute the correct edges
        correct_edges = gold_edges & sys_edges
        
        # Accumulate
        current_n_correct_edges += len(correct_edges)
        current_n_predicted_edges += len(sys_edges)
        current_n_actual_edges += len(gold_edges)
        
        if len(correct_edges) != len(gold_edges):
          current_n_correct_sequences = 0
        if len(gold_edges | sys_edges):
          current_n_actual_tokens += 1
          if len(correct_edges) == len(gold_edges):
            current_n_correct_tokens += 1
        current_n_tokens += 1
      gold_line = gf.readline()
  
  if current_n_tokens:
    n_correct_edges += current_n_correct_edges
    n_predicted_edges += current_n_predicted_edges
    n_actual_edges += current_n_actual_edges
    
    n_correct_tokens += current_n_correct_tokens
    n_actual_tokens += current_n_actual_tokens
    n_tokens += current_n_tokens
    
    n_correct_sequences += current_n_correct_sequences
    n_sequences += 1
  
  Accuracy = namedtuple('Accuracy', ['precision', 'recall', 'F1', 'tok_acc', 'seq_acc'])
  precision = n_correct_edges / (n_predicted_edges + 1e-12)
  recall = n_correct_edges / (n_actual_edges + 1e-12)
  F1 = 2 * precision * recall / (precision + recall + 1e-12)
  tok_acc = n_correct_tokens / (n_actual_tokens + 1e-12)
  seq_acc = n_correct_sequences / (n_sequences + 1e-12)
  accuracy = Accuracy(precision, recall, F1, tok_acc, seq_acc)
  
  return accuracy
  
#===============================================================
def semdep_cleanup(hyperparam_dict):
  """"""
  
  if not hyperparam_dict['SemrelGraphTokenVocab']['factorized']:
    hyperparam_dict['SemrelGraphTokenVocab']['loss_interpolation'] = None
    if hyperparam_dict['SemrelGraphTokenVocab']['decomposition_level'] in ('0', 0):
      hyperparam_dict['SemrelGraphTokenVocab']['diagonal'] = None
    for option in hyperparam_dict['SemheadGraphIndexVocab']:
      hyperparam_dict['SemheadGraphIndexVocab'][option] = hyperparam_dict['SemrelGraphTokenVocab'][option]
  else:
    if hyperparam_dict['SemheadGraphIndexVocab']['decomposition_level'] in ('0', 0):
      hyperparam_dict['SemheadGraphIndexVocab']['diagonal'] = None
  if hyperparam_dict['SemrelGraphTokenVocab']['decomposition_level'] in ('0', 0):
    hyperparam_dict['Network']['recur_size'] = np.clip(hyperparam_dict['Network']['recur_size'], 10, 450)
    hyperparam_dict['SemrelGraphTokenVocab']['hidden_size'] = None
    hyperparam_dict['SemrelGraphTokenVocab']['n_layers'] = None
    hyperparam_dict['SemrelGraphTokenVocab']['hidden_func'] = None
  if hyperparam_dict['SemheadGraphIndexVocab']['decomposition_level'] in ('0', 0):
    hyperparam_dict['SemheadGraphIndexVocab']['hidden_size'] = None
    hyperparam_dict['SemheadGraphIndexVocab']['n_layers'] = None
    hyperparam_dict['SemheadGraphIndexVocab']['hidden_func'] = None
  return hyperparam_dict
