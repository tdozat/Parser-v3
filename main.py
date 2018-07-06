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
#from hpo.evals.syndep_eval import evaluate_tokens
from hpo.evals.conll18_eval import evaluate

section_names = set()
with codecs.open(os.path.join('config', 'defaults.cfg')) as f:
  section_regex = re.compile('\[(.*)\]')
  for line in f:
    match = section_regex.match(line)
    if match:
      section_names.add(match.group(1))

#-------------------------------------------------------------
def resolve_network_dependencies(config, network_class, network_list, networks):
  if network_list in ('None', ''):
    return set(), networks
  else:
    network_list = network_list.split(':')
    if network_class not in networks:
      for _network_class in network_list:
        config_file = os.path.join(config.get('DEFAULT', _network_class + '_dir'), 'config.cfg')
        _config = Config(config_file=config_file)
        _network_list = _config.get(_network_class, 'input_network_classes')
        input_networks, networks = resolve_network_dependencies(_config, _network_class, _network_list, networks)
        NetworkClass = getattr(parser, _network_class)
        networks[_network_class] = NetworkClass(input_networks=input_networks, config=config)
    return set(networks[_network_class] for _network_class in network_list), networks
#-------------------------------------------------------------

# TODO make all DEFAULT options as regular flags
#***************************************************************
# Set up the argument parser
def main():
  """ --section_name opt1=value1 opt2=value2 opt3=value3 """
  
  argparser = ArgumentParser('Network')
  argparser.add_argument('--save_metadir')
  argparser.add_argument('--save_dir')
  subparsers = argparser.add_subparsers()
  
  # set up the training parser
  train_parser = subparsers.add_parser('train')
  train_parser.set_defaults(action=train)
  train_parser.add_argument('network_class')
  train_parser.add_argument('--force', action='store_true')
  train_parser.add_argument('--noscreen', action='store_true')
  train_parser.add_argument('--load', action='store_true')
  train_parser.add_argument('--config_file', default='')
  for section_name in section_names:
    train_parser.add_argument('--'+section_name, nargs='+')

  hpo_parser = subparsers.add_parser('hpo')
  hpo_parser.set_defaults(action=hpo)
  hpo_parser.add_argument('network_class')
  hpo_parser.add_argument('--noscreen', action='store_true')
  hpo_parser.add_argument('--config_file', default='')
  hpo_parser.add_argument('--rand_file', default='hpo/config/default.csv')
  hpo_parser.add_argument('--eval_metric', default='LAS')
  for section_name in section_names:
    hpo_parser.add_argument('--'+section_name, nargs='+')

  # set up the hpo parser
  # set up the inference parser
  run_parser = subparsers.add_parser('run')
  run_parser.set_defaults(action=run)
  run_parser.add_argument('conllu_files', nargs='+')
  run_parser.add_argument('--output_dir')
  run_parser.add_argument('--output_filename')
  for section_name in section_names:
    run_parser.add_argument('--'+section_name, nargs='+')
    
  # parse the arguments
  kwargs = vars(argparser.parse_args())
  kwargs.pop('action')(**kwargs)
  return

#===============================================================
# Train
def train(**kwargs):
  """"""
  
  # Get the special arguments
  load = kwargs.pop('load')
  force = kwargs.pop('force')
  noscreen = kwargs.pop('noscreen')
  save_dir = kwargs.pop('save_dir')
  save_metadir = kwargs.pop('save_metadir')
  network_class = kwargs.pop('network_class')
  config_file = kwargs.pop('config_file')
  
  # Get the cl-defined options
  kwargs = {key: value for key, value in six.iteritems(kwargs) if value is not None}
  for section, values in six.iteritems(kwargs):
    if section in section_names:
      values = [value.split('=', 1) for value in values]
      kwargs[section] = {opt: value for opt, value in values}
  if 'DEFAULT' not in kwargs:
    kwargs['DEFAULT'] = {}
  kwargs['DEFAULT']['network_class'] = network_class
    
  # Figure out the save_directory
  if save_metadir is not None:
    kwargs['DEFAULT']['save_metadir'] = save_metadir
  if save_dir is None:
    save_dir = Config(config_file=config_file, **kwargs).get('DEFAULT', 'save_dir')
  
  # If not loading, ask the user if they want to overwrite the directory
  if not load and os.path.isdir(save_dir):
    if not force:
      input_str = ''
      while input_str not in ('y', 'n', 'yes', 'no'):
        input_str = input('{} already exists. It will be deleted if you continue. Do you want to proceed? [Y/n] '.format(save_dir)).lower()
      if input_str in ('n', 'no'):
        print()
        sys.exit(0)
    shutil.rmtree(save_dir)
  
  # If the save_dir wasn't overwritten, load its config_file
  if os.path.isdir(save_dir):
    config_file = os.path.join(save_dir, 'config.cfg')
  else:
    os.makedirs(save_dir)
    os.system('git rev-parse HEAD >> {}'.format(os.path.join(save_dir, 'HEAD')))
  
  kwargs['DEFAULT']['save_dir'] = save_dir
  config = Config(config_file=config_file, **kwargs)
  network_list = config.get(network_class, 'input_network_classes')
  if not load:
    with open(os.path.join(save_dir, 'config.cfg'), 'w') as f:
      config.write(f)
  input_networks, networks = resolve_network_dependencies(config, network_class, network_list, {})
  NetworkClass = getattr(parser, network_class)
  network = NetworkClass(input_networks=input_networks, config=config)
  network.train(load=load, noscreen=noscreen)
  return

#===============================================================
# HPO
def hpo(**kwargs):
  """"""
  
  # Get the special arguments
  noscreen = kwargs.pop('noscreen')
  save_dir = kwargs.pop('save_dir')
  save_metadir = kwargs.pop('save_metadir')
  network_class = kwargs.pop('network_class')
  config_file = kwargs.pop('config_file')
  rand_file = kwargs.pop('rand_file')
  eval_metric = kwargs.pop('eval_metric')
  
  # Get the cl-defined options
  kwargs = {key: value for key, value in six.iteritems(kwargs) if value is not None}
  for section, values in six.iteritems(kwargs):
    if section in section_names:
      values = [value.split('=', 1) for value in values]
      kwargs[section] = {opt: value for opt, value in values}
  if 'DEFAULT' not in kwargs:
    kwargs['DEFAULT'] = {}
  kwargs['DEFAULT']['network_class'] = network_class
    
  # Figure out the save_directory
  if save_metadir is not None:
    kwargs['DEFAULT']['save_metadir'] = save_metadir
  if save_dir is None:
    save_dir = Config(**kwargs).get('DEFAULT', 'save_dir')
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  
  # Get the randomly generated options and possibly add them
  #-------------------------------------------------------------
  lang = kwargs['DEFAULT']['LANG']
  treebank = kwargs['DEFAULT']['TREEBANK']
  lc = kwargs['DEFAULT']['LC']
  tb = kwargs['DEFAULT']['TB']
  base = 'data/CoNLL18/UD_{}-{}/{}_{}-ud-dev.conllu'.format(lang, treebank, lc, tb)
  def eval_func(save_dir):
    return evaluate(base, os.path.join(save_dir, 'parsed', base), eval_metric)
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
  
  # If not loading, ask the user if they want to overwrite the directory
  if os.path.isdir(save_dir):
    print()
    sys.exit(0)
  else:
    os.mkdir(save_dir)
    os.system('git rev-parse HEAD >> {}'.format(os.path.join(save_dir, 'HEAD')))
  
  kwargs['DEFAULT']['save_dir'] = save_dir
  config = Config(config_file=config_file, **kwargs)
  network_list = config.get(network_class, 'input_network_classes')
  with open(os.path.join(save_dir, 'config.cfg'), 'w') as f:
    config.write(f)
  input_networks, networks = resolve_network_dependencies(config, network_class, network_list, {})
  NetworkClass = getattr(parser, network_class)
  network = NetworkClass(input_networks=input_networks, config=config)
  network.train(noscreen=noscreen)
  return

#===============================================================
def run(**kwargs):
  """"""

  # Get the special arguments
  save_dir = kwargs.pop('save_dir')
  save_metadir = kwargs.pop('save_metadir')
  conllu_files = kwargs.pop('conllu_files')
  output_dir = kwargs.pop('output_dir')
  output_filename = kwargs.pop('output_filename')

  # Get the cl-defined options
  kwargs = {key: value for key, value in six.iteritems(kwargs) if value is not None}
  for section, values in six.iteritems(kwargs):
    if section in section_names:
      values = [value.split('=', 1) for value in values]
      kwargs[section] = {opt: value for opt, value in values}
  if 'DEFAULT' not in kwargs:
    kwargs['DEFAULT'] = {}
    
  # Figure out the save_directory
  if save_metadir is not None:
    kwargs['DEFAULT']['save_metadir'] = save_metadir
  if save_dir is None:
    save_dir = Config(**kwargs).get('DEFAULT', 'save_dir')
  config_file = os.path.join(save_dir, 'config.cfg')
  kwargs['DEFAULT']['save_dir'] = save_dir

  config = Config(defaults_file='', config_file=config_file, **kwargs)
  network_class = config.get('DEFAULT', 'network_class')
  network_list = config.get(network_class, 'input_network_classes')
  input_networks, networks = resolve_network_dependencies(config, network_class, network_list, {})
  NetworkClass = getattr(parser, network_class)
  network = NetworkClass(input_networks=input_networks, config=config)
  network.parse(conllu_files, output_dir=output_dir, output_filename=output_filename)
  return

#***************************************************************
if __name__ == '__main__':
  main()
