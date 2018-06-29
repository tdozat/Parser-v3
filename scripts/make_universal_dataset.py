from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import glob

conll18_dir = 'data/CoNLL18/'
unilang_dir = 'data/CoNLL18/UD_Unilang-1000'
os.makedirs(unilang_dir, exist_ok=True)

with codecs.open(os.path.join(unilang_dir, 'uni_1000-ud-train.conllu'), 'w') as unitrain_conllu,\
     codecs.open(os.path.join(unilang_dir, 'uni_1000-ud-train.txt'), 'w') as unitrain_txt,\
     codecs.open(os.path.join(unilang_dir, 'uni_1000-ud-dev.conllu'), 'w') as unidev_conllu,\
     codecs.open(os.path.join(unilang_dir, 'uni_1000-ud-dev.txt'), 'w') as unidev_txt:
  for lang_dir in os.listdir(conll18_dir):
    conll_lang_dir = os.path.join(conll18_dir, lang_dir)
    if conll_lang_dir != unilang_dir and not (conll_lang_dir.endswith('-MG') or conll_lang_dir.endswith('ITTB_XV')):
      print(conll_lang_dir)
      # Train conllu
      train_conllu = glob.glob(os.path.join(conll_lang_dir, '*-train.conllu'))
      if len(train_conllu) == 0:
        continue
      unitrain_conllu.write('# {}\n'.format(lang_dir))
      with codecs.open(train_conllu.pop(), encoding='utf-8') as lang_train_conllu:
        cur_sent = 0
        for line in lang_train_conllu:
          if line.rstrip():
            unitrain_conllu.write(line)
            if line.startswith('# text = '):
              unitrain_txt.write(line[9:])
          else:
            unitrain_conllu.write(line)
            cur_sent += 1
          if cur_sent >= 1000:
            break
      unitrain_txt.write('\n')

      # dev_conllu
      dev_conllu = glob.glob(os.path.join(conll_lang_dir, '*-dev.conllu'))
      if len(dev_conllu) == 0:
        continue
      assert len(dev_conllu) == 1
      unidev_conllu.write('# {}\n'.format(lang_dir))
      with codecs.open(dev_conllu.pop(), encoding='utf-8') as lang_dev_conllu:
        cur_sent = 0
        for line in lang_dev_conllu:
          if line.rstrip():
            unidev_conllu.write(line)
            if line.startswith('# text = '):
              unidev_txt.write(line[9:])
          else:
            unidev_conllu.write(line)
            cur_sent += 1
          if cur_sent >= 100:
            break
      unidev_txt.write('\n')
