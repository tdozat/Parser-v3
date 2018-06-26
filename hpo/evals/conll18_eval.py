from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scripts.conll18_ud_eval as ud_eval

def evaluate(gold_filename, sys_filename, metric):
  """"""
  
  gold_conllu_file = ud_eval.load_conllu_file(gold_filename)
  sys_conllu_file = ud_eval.load_conllu_file(sys_filename)
  evaluation = evaluate(gold_conllu_file, sys_conllu_file)
  return evaluation[metric].f1
  
