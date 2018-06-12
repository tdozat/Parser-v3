#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Timothy Dozat
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

from collections import defaultdict

#===============================================================
def to_dumb(conllu_filename):
  """"""
  
  with open(conllu_filename) as f:
    with open(conllu_filename+'.sdp', 'w') as g:
      buff = []
      for line in f:
        line = line.strip('\n')
        line = line.split('\t')
        if len(line) == 10:
          # Add it to the buffer
          buff.append(line)
        elif buff:
          # Process the buffer
          words = []
          top = set()
          preds = set()
          graph = defaultdict(dict)
          for i, line in enumerate(buff):
            words.append( [line[0], line[1], line[2], line[4]] )
            if line[8] != '_':
              nodes = line[8].split('|')
              for node in nodes:
                node = node.split(':', 1)
                node[0] = int(node[0])
                if node[0] == 0:
                  top.add(i)
                else:
                  preds.add(node[0])
                  graph[i][node[0]] = node[1]
          sorted_preds = sorted(list(preds))
          for i, word in enumerate(words):
            # Add the top
            if i in top:
              word.append('+')
            else:
              word.append('-')
            # Add the preds
            if i in preds:
              word.append('+')
            else:
              word.append('-')
            # Add the graph
            for pred in sorted_preds:
              if pred in graph[i]:
                word.append(graph[i][pred])
              else:
                word.append('_')
            g.write('\t'.join(word) + '\n')
          g.write('\n')
          buff = []

#***************************************************************
if __name__ == '__main__':
  """"""
  
  import sys
  to_dumb(sys.argv[1])
