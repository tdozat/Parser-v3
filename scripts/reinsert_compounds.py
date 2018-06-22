from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function 
 
import os 
import sys 
import codecs 
 
gold_file = sys.argv[1] 
system_file = sys.argv[2] 
 
system_lines = [] 
 
with codecs.open(system_file, encoding='utf-8') as fin: 
  for system_line in fin: 
    system_lines.append(system_line) 
 
with codecs.open(gold_file, encoding='utf-8') as fin: 
  with codecs.open(system_file, 'w', encoding='utf-8') as fout: 
    i = 0 
    for gold_line in fin: 
      gold_line = gold_line.strip() 
 
      if len(gold_line) == 0: 
        fout.write(system_lines[i]) 
        i += 1 
        continue 
 
      if gold_line[0] == '#': 
        fout.write('{}\n'.format(gold_line)) 
        continue 
 
      splitline = gold_line.split('\t') 
      if '.' in splitline[0]: 
        fout.write('{}\n'.format(gold_line)) 
        continue 
 
      if '-' in splitline[0]: 
        fout.write('{}\n'.format(gold_line)) 
        continue 

      fout.write(system_lines[i])
      i += 1
