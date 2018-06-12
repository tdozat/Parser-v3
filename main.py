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

import re
import os
import shutil
import sys
import time
import six
from six.moves import input
import codecs
from argparse import ArgumentParser

from parser.config import Config
import parser 
from hpo import MVGHPO
from hpo.evals.syndep_eval import evaluate_tokens

#***************************************************************
#===============================================================
# Set up the argument parser
def main():
  """ --section_name opt1=value1 opt2=value2 opt3=value3 """
  
  argparser = ArgumentParser('Network')
  argparser.add_argument('--save_dir', required=True)
  subparsers = argparser.add_subparsers()
  section_names = set()
  with codecs.open(os.path.join('config', 'defaults.cfg')) as f:
    section_regex = re.compile('\[(.*)\]')
    for line in f:
      match = section_regex.match(line)
      if match:
        section_names.add(match.group(1))
  
  # set up the training parser
  train_parser = subparsers.add_parser('train')
  train_parser.set_defaults(action=train)
  train_parser.add_argument('network_class')
  train_parser.add_argument('--load', action='store_true')
  train_parser.add_argument('--config_file', default='')
  train_parser.add_argument('--randomize', nargs='?', const='hpo/config/default.csv')
  for section_name in section_names:
    train_parser.add_argument('--'+section_name, nargs='+')
    
  # parse the arguments
  kwargs = vars(argparser.parse_args())
  action = kwargs.pop('action')
  save_dir = kwargs.pop('save_dir')
  rand_file = kwargs.pop('randomize')
  # Get the cl-defined options
  kwargs = {key: value for key, value in six.iteritems(kwargs) if value is not None}
  for section, values in six.iteritems(kwargs):
    if section in section_names:
      values = [value.split('=', 1) for value in values]
      kwargs[section] = {opt: value for opt, value in values}
  # Get the randomly generated options and possibly add them
  if rand_file is not None:
    #-------------------------------------------------------------
    for arg in kwargs['DEFAULT']:
      if arg.startswith('LANG'):
        lang = arg.split('=')[1]
      elif arg.startswith('TREEBANK'):
        treebank = arg.split('=')[1]
      elif arg.startswith('LC'):
        lc = arg.split('=')[1]
      elif arg.startswith('TB'):
        tb = arg.split('=')[1]
    base = 'data/CoNLL18/UD_{}-{}/{}_{}-ud-dev.conllu'.format(lang, treebank, lc, tb)
    def eval_func(save_dir):
      return evaluate_tokens(base, os.path.join(save_dir, 'parsed', base))
    #-------------------------------------------------------------
    rargs = next(MVGHPO(rand_file, save_dir, eval_func=eval_func))
    for section in rargs:
      if section not in kwargs:
        kwargs[section] = rargs[section]
      else:
        for option, value in six.iteritems(rargs[section]):
          if option not in kwargs[section]:
            kwargs[section][option] = value
    save_dir = os.path.join(save_dir, str(int(time.time()*100000)))
  action(save_dir, **kwargs)
  return

#===============================================================
# Train
def train(save_dir, **kwargs):
  """"""
  
  load = kwargs.pop('load')
  if not load and os.path.isdir(save_dir):
    input_str = ''
    while input_str not in ('y', 'n', 'yes', 'no'):
      input_str = input('Save directory already exists. It will be deleted if you continue. Do you want to proceed? [Y/n] ').lower()
    if input_str in ('n', 'no'):
      print()
      sys.exit(0)
    else:
      shutil.rmtree(save_dir)
  
  if os.path.isdir(save_dir):
    config_file = os.path.join(save_dir, 'config.cfg')
  else:
    os.mkdir(save_dir)
    config_file = kwargs.pop('config_file', '')
  
  if 'DEFAULT' not in kwargs:
    kwargs['DEFAULT'] = {}
  kwargs['DEFAULT']['save_dir'] = save_dir
  network_class = getattr(parser, kwargs.pop('network_class'))
  config = Config(config_file=config_file, **kwargs)
  network = network_class(config=config)
  if not load:
    with open(os.path.join(save_dir, 'config.cfg'), 'w') as f:
      config.write(f)
  network.train(load=load)
  return

#***************************************************************
if __name__ == '__main__':
  main()
