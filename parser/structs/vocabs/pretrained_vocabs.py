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
import six

import os
import codecs
import zipfile
import gzip
try:
  import lzma
except:
  try:
    from backports import lzma
  except:
    import warnings
    warnings.warn('Install backports.lzma for xz support')
try:
  import cPickle as pkl
except ImportError:
  import pickle as pkl
from collections import Counter

import numpy as np
import tensorflow as tf
 
from parser.structs.vocabs.base_vocabs import SetVocab
from . import conllu_vocabs as cv
from parser.neural import embeddings

#***************************************************************
# TODO maybe change self.name to something more like self._save_str?
# Ideally there should be Word2vecVocab, GloveVocab, FasttextVocab,
# each with their own _save_str
class PretrainedVocab(SetVocab):
  """"""
  
  #=============================================================
  def __init__(self, pretrained_file=None, name=None, config=None):
    """"""
    
    if (pretrained_file is None) != (name is None):
      raise ValueError("You can't pass in a value for only one of pretrained_file and name to PretrainedVocab.__init__")
    
    if pretrained_file is None:
      pretrained_file = config.getstr(self, 'pretrained_file')
      name = config.getstr(self, 'name')
    super(PretrainedVocab, self).__init__(config=config)
    self._pretrained_file = pretrained_file
    self._name = name
    self.variable = None
    return
  
  #=============================================================
  def get_input_tensor(self, embed_keep_prob=None, variable_scope=None, reuse=True):
    """"""
    
    # Default override
    embed_keep_prob = embed_keep_prob or self.embed_keep_prob
    
    with tf.variable_scope(variable_scope or self.field):
      if self.variable is None:
        with tf.device('/cpu:0'):
          self.variable = tf.Variable(self.embeddings, name=self.name+'Embeddings', trainable=False)
          tf.add_to_collection('non_save_variables', self.variable)
      layer = embeddings.pretrained_embedding_lookup(self.variable, self.linear_size,
                                                     self.placeholder,
                                                     name=self.name,
                                                     reuse=reuse)
      if embed_keep_prob < 1:
        layer = self.drop_func(layer, embed_keep_prob)
    return layer
    
  #=============================================================
  def count(self, *args):
    """"""
    
    max_embed_count = self.max_embed_count
    if self.pretrained_file.endswith('.zip'):
      open_func = zipfile.Zipfile
      kwargs = {}
    elif self.pretrained_file.endswith('.gz'):
      open_func = gzip.open
      kwargs = {}
    elif self.pretrained_file.endswith('.xz'):
      open_func = lzma.open
      kwargs = {'errors': 'ignore'}
    else:
      open_func = codecs.open
      kwargs = {'errors': 'ignore'}
    
    cur_idx = len(self.special_tokens)
    tokens = []
    # Determine the dimensions of the embedding matrix
    with open_func(self.pretrained_file, 'rb') as f:
      reader = codecs.getreader('utf-8')(f, **kwargs)
      first_line = reader.readline().rstrip().split(' ')
      if len(first_line) == 2: # It has a header that gives the dimensions
        has_header = True
        shape = [int(first_line[0])+cur_idx, int(first_line[1])]
      else: # We have to compute the dimensions ourself
        has_header = False
        for line_num, line in enumerate(reader):
          pass
        shape = [cur_idx+line_num+1, len(line.split())-1]
      shape[0] = min(shape[0], max_embed_count+cur_idx) if max_embed_count else shape[0]
      embeddings = np.zeros(shape, dtype=np.float32)
      
      # Fill in the embedding matrix
      #with open_func(self.pretrained_file, 'rt', encoding='utf-8') as f:
      with open_func(self.pretrained_file, 'rb') as f:
        for line_num, line in enumerate(f):
          if line_num:
            if cur_idx < shape[0]:
              line = line.rstrip()
              if line:
                line = line.decode('utf-8', errors='ignore').split(' ')
                embeddings[cur_idx] = line[1:]
                tokens.append(line[0])
                self[line[0]] = cur_idx
                cur_idx += 1
            else:
              break
    
    self._embed_size = shape[1]
    self._tokens = tokens
    self._embeddings = embeddings
    self.dump()
    return True

  #=============================================================
  def dump(self):
    if self.save_as_pickle:
      with open(self.vocab_savename, 'wb') as f:
        pkl.dump((self._tokens, self._embeddings), f, protocol=pkl.HIGHEST_PROTOCOL)
    return

  #=============================================================
  def load(self):
    """"""

    dump = None
    if os.path.exists(self.vocab_savename):
      vocab_filename = self.vocab_savename
      dump = False
    elif self.vocab_loadname and os.path.exists(self.vocab_loadname):
      vocab_filename = self.vocab_loadname
      dump = True
    else:
      self._loaded = False
      return False

    with open(vocab_filename, 'rb') as f:
      self._tokens, self._embeddings = pkl.load(f, encoding='utf-8', errors='ignore')
    cur_idx = len(self.special_tokens)
    for token in self._tokens:
      self[token] = cur_idx
      cur_idx += 1
    self._embedding_size = self._embeddings.shape[1]
    if dump:
      self.dump()
    self._loaded = True
    return True

  #=============================================================
  @property
  def pretrained_file(self):
    return self._config.getstr(self, 'pretrained_file')
  @property
  def vocab_loadname(self):
    return self._config.getstr(self, 'vocab_loadname')
  @property
  def vocab_savename(self):
    return os.path.join(self.save_dir, self.field + '-' + self.name + '.pkl')
    #return os.path.splitext(self.pretrained_file)[0] + '.pkl'
  @property
  def name(self):
    return self._name
  @property
  def max_embed_count(self):
    return self._config.getint(self, 'max_embed_count')
  @property
  def embeddings(self):
    return self._embeddings
  @property
  def embed_keep_prob(self):
    return self._config.getfloat(self, 'max_embed_count')
  @property
  def embed_size(self):
    return self._embed_size
  @property
  def save_as_pickle(self):
    return self._config.getboolean(self, 'save_as_pickle')
  @property
  def linear_size(self):
    return self._config.getint(self, 'linear_size')
  
#***************************************************************
class FormPretrainedVocab(PretrainedVocab, cv.FormVocab):
  pass
class LemmaPretrainedVocab(PretrainedVocab, cv.LemmaVocab):
  pass
class UPOSPretrainedVocab(PretrainedVocab, cv.UPOSVocab):
  pass
class XPOSPretrainedVocab(PretrainedVocab, cv.XPOSVocab):
  pass
class DeprelPretrainedVocab(PretrainedVocab, cv.DeprelVocab):
  pass
