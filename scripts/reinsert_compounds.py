from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function 
 
import os 
import codecs 
 
#*************************************************************** 
def remove_compounds(system_file):
  system_lines = [] 

  with codecs.open(system_file, encoding='utf-8') as fin: 
    for system_line in fin: 
      system_lines.append(system_line) 
  with codecs.open(system_file, 'w', encoding='utf-8') as fout: 
    for system_line in system_lines:
      system_line = system_line.rstrip()
      if not system_line:
        fout.write('\n')
        continue

      if system_line.startswith('#'):
        continue

      system_splitline = system_line.split('\t')
      if '-' in system_splitline[0] or '.' in system_splitline[0]:
        continue
      
      fout.write(system_line+'\n')
  return

#===============================================================
def reinsert_compounds(gold_file, system_file):
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
          if system_lines[i][0] == '#':
            i += 1
          continue 
   
        gold_splitline = gold_line.split('\t') 
        system_splitline = system_lines[i].split('\t')
        if '.' in gold_splitline[0]: 
          fout.write('{}\n'.format(gold_line)) 
          if '.' in system_splitline[0]:
            i += 1
          continue 
   
        if '-' in gold_splitline[0]: 
          fout.write('{}\n'.format(gold_line)) 
          if '-' in system_splitline[0]:
            i += 1
          continue 
  
        fout.write(system_lines[i])
        i += 1
  return

#=============================================================== 
def main(gold_file, system_file):
  reinsert_compounds(gold_file, system_file)
 
#*************************************************************** 
if __name__ == '__main__':
  
  import sys 
   
  if sys.argv[1] == 'remove':
    remove_compounds(sys.argv[2])
  else:
    reinsert_compounds(sys.argv[1], sys.argv[2])
