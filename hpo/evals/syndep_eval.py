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
import re
from numpy import nan

try:
  import cPickle as pkl
except ImportError:
  import pickle as pkl

#===============================================================
def evaluate_tokens(gold_file, sys_file, labeled=False, force=False):
  """"""
  
  if not os.path.exists(sys_file):
    return nan
  
  if os.path.exists(sys_file + '-score.pkl') and not force:
    with open(sys_file + '-score.pkl', 'rb') as f:
      return pkl.load(f)
  
  n_correct_tokens = 0
  n_tokens = 0
  
  current_n_correct_tokens = 0
  current_n_tokens = 0
  
  with codecs.open(gold_file, encoding='utf-8') as gf,\
       codecs.open(sys_file, encoding='utf-8') as sf:
    gold_line = gf.readline()
    while gold_line:
      while gold_line.startswith('#') or re.match('[0-9]*\.', gold_line):
        if current_n_tokens:
          n_correct_tokens += current_n_correct_tokens
          n_tokens += current_n_tokens
        
        current_n_correct_tokens = 0
        current_n_tokens = 0
        
        gold_line = gf.readline()
      
      if gold_line != '\n': 
        sys_line = sf.readline()
        while sys_line.startswith('#') or sys_line == '\n':
          sys_line = sf.readline()
        
        try:
          assert gold_line[0] == sys_line[0]
        except IndexError:
          print(gold_line, sys_line)
          raise
        gold_line = gold_line.rstrip().split('\t')
        sys_line = sys_line.rstrip().split('\t')
        
        # Compute the gold edges
        gold_head = gold_line[6]
        gold_label = gold_line[7]
        
        # Compute the sys edges
        sys_head = sys_line[6]
        sys_label = sys_line[7]
        
        # Accumulate
        if labeled:
          current_n_correct_tokens += (gold_head == sys_head) and (gold_label == sys_label)
        else:
          current_n_correct_tokens += (gold_head == sys_head)
        current_n_tokens += 1
      gold_line = gf.readline()
  
  if current_n_tokens:
    n_correct_tokens += current_n_correct_tokens
    n_tokens += current_n_tokens
  
  score = n_correct_tokens / n_tokens if n_tokens else nan
  with open(sys_file + '-score.pkl', 'wb') as f:
    pkl.dump(score, f)
  return score

#***************************************************************
if __name__ == '__main__':
  """"""
  
  print(evaluate_tokens('data/test/gold/conll17-ud-test-2017-05-09/en.conllu', 'saves/test/parsed/data/test/gold/conll17-ud-test-2017-05-09/en.conllu'))
