from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import codecs
import glob
import sys

lang_dir = os.path.normpath(sys.argv[1])

lang_xv_dir = lang_dir+'_XV'
os.makedirs(lang_xv_dir, exist_ok=True)
train_conllu = glob.glob(os.path.join(lang_dir, '*-ud-train.conllu')).pop(0)
train_txt = glob.glob(os.path.join(lang_dir, '*-ud-train.txt')).pop(0)
basename = os.path.splitext(os.path.basename(train_conllu))[0]
lc_tb = re.match('(.*)-ud-train', basename).group(1)
xv_tb = lc_tb + '_xv'

with codecs.open(os.path.join(lang_xv_dir, xv_tb+'-ud-train.conllu'), 'w') as xv_train_conllu,\
     codecs.open(os.path.join(lang_xv_dir, xv_tb+'-ud-train.txt'), 'w') as xv_train_txt,\
     codecs.open(os.path.join(lang_xv_dir, xv_tb+'-ud-dev.conllu'), 'w') as xv_dev_conllu,\
     codecs.open(os.path.join(lang_xv_dir, xv_tb+'-ud-dev.txt'), 'w') as xv_dev_txt:
  with codecs.open(train_conllu, encoding='utf-8') as lang_train_conllu:
    cur_sent = 0
    for line in lang_train_conllu:
      cur_conllu_file, cur_txt_file = (xv_train_conllu, xv_train_txt) if (cur_sent % 8) else (xv_dev_conllu, xv_dev_txt)
      if line.rstrip():
        cur_conllu_file.write(line)
        if line.startswith('# text = '):
          cur_txt_file.write(line[9:])
      else:
        cur_conllu_file.write(line)
        cur_sent += 1

