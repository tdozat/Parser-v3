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

import numpy as np
import tensorflow as tf

from .base_vocabs import BaseVocab
from . import conllu_vocabs as cv
from . import token_vocabs as tv
from . import pretrained_vocabs as pv
from . import subtoken_vocabs as sv

from parser.neural import embeddings

#***************************************************************
class Multivocab(BaseVocab, list): 
  """"""
  
  _token_vocab_class = None
  _subtoken_vocab_class = None
  _pretrained_vocab_class = None
  
  #=============================================================
  def __init__(self, config=None):
    """"""
    
    super(Multivocab, self).__init__(config=config)
    list.__init__(self)
    
    # Set up the frequent-token vocab
    use_token_vocab = config.getboolean(self, 'use_token_vocab')
    if use_token_vocab:
      token_vocab = self._token_vocab_class(config=config)
      self.append(token_vocab)
    else:
      token_vocab = None
    
    # Set up the character-based vocab
    use_subtoken_vocab = config.getboolean(self, 'use_subtoken_vocab')
    if use_subtoken_vocab:
      subtoken_vocab = self._subtoken_vocab_class(config=config)
      self.append(subtoken_vocab)
    else:
      subtoken_vocab = None
    
    # Set up the pretrained vocab(s)
    use_pretrained_vocab = config.getboolean(self, 'use_pretrained_vocab')
    pretrained_files = config.getlist(self, 'pretrained_files')
    names = config.getlist(self, 'names')
    if use_pretrained_vocab:
      if pretrained_files is None:
        pretrained_vocabs = [self._pretrained_vocab_class(config=config)]
      else:
        pretrained_vocabs = [self._pretrained_vocab_class(pretrained_file=pretrained_file, pretrained_name=pretrained_name, config=config) for pretrained_file, name in zip(pretrained_files, names)]
      self.extend(pretrained_vocabs)
    else:
      pretrained_vocabs = []
      
    # Set the special tokens
    for base_special_token in self[0].base_special_tokens:
      self.__dict__[base_special_token.upper()+'_STR'] = self[0].__dict__[base_special_token.upper()+'_STR']
      self.__dict__[base_special_token.upper()+'_IDX'] = tuple(vocab.__dict__[base_special_token.upper()+'_IDX'] for vocab in self)
    
    self._token_vocab = token_vocab
    self._subtoken_vocab = subtoken_vocab
    self._pretrained_vocabs = pretrained_vocabs
    return
  
  #=============================================================
  def load(self):
    """"""
    
    status = True
    for vocab in self:
      status = (vocab.load() if hasattr(vocab, 'load') else True) and status
    return status
  
  #=============================================================
  def count(self, train_conllus):
    """"""
    
    status = True
    for vocab in self:
      status = (vocab.count(train_conllus) if hasattr(vocab, 'count') else True) and status
    return status
  
  #=============================================================
  def get_input_tensor(self, nonzero_init=False, reuse=True):
    """"""
    
    embed_keep_prob = 1 if reuse else self.embed_keep_prob
    #if self.combine_func != embeddings.concat:
    #  assert len(set([vocab.embed_size for vocab in self])) == 1, "Unless Multivocab.combine_func is set to 'concat', all vocabs must have the same 'embed_size'"
    
    with tf.variable_scope(self.field):
      input_tensors = []
      if self._pretrained_vocabs:
        with tf.variable_scope('Pretrained') as variable_scope:
          input_tensors.extend([pretrained_vocab.get_input_tensor(embed_keep_prob=1., variable_scope=variable_scope, reuse=reuse) for pretrained_vocab in self._pretrained_vocabs])
          nonzero_init = False
      
      if self._subtoken_vocab is not None:
        with tf.variable_scope('Subtoken') as variable_scope:
          input_tensors.append(self._subtoken_vocab.get_input_tensor(nonzero_init=nonzero_init, embed_keep_prob=1., variable_scope=variable_scope, reuse=reuse))
          nonzero_init = False
      
      if self._token_vocab is not None:
        with tf.variable_scope('Token') as variable_scope:
          input_tensors.append(self._token_vocab.get_input_tensor(nonzero_init=nonzero_init, embed_keep_prob=1., variable_scope=variable_scope, reuse=reuse))
      
      layer = self.combine_func(input_tensors, embed_keep_prob=embed_keep_prob, drop_func=self.drop_func)
    return layer
  
  #=============================================================
  def add(self, token):
    """"""
    
    return tuple(vocab.add(token) for vocab in self)
  
  #=============================================================
  def index(self, token):
    """"""
    
    return tuple(vocab.index(token) for vocab in self)
  
  #=============================================================
  def token(self, index):
    """"""
    
    return self[0].token(index)
  
  #=============================================================
  def set_placeholders(self, indices, feed_dict={}):
    """"""
    
    for i, vocab in enumerate(self):
      vocab.set_placeholders(indices[:,:,i], feed_dict=feed_dict)
    return feed_dict
  
  #=============================================================
  def get_root(self):
    """"""
    
    return self.ROOT_STR
  
  #=============================================================
  def open(self):
    for vocab in self:
      vocab.open()
    return self
  
  def close(self):
    for vocab in self:
      vocab.close()
    return
  
  #=============================================================
  @property
  def depth(self):
    return len(self)
  @property
  def combine_func(self):
    combine_func = self._config.getstr(self, 'combine_func')
    if hasattr(embeddings, combine_func):
      return getattr(embeddings, combine_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(embeddings.__name__, combine_func))
  @property
  def drop_func(self):
    drop_func = self._config.getstr(self, 'drop_func')
    if hasattr(embeddings, drop_func):
      return getattr(embeddings, drop_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(embeddings.__name__, drop_func))
  @property
  def embed_keep_prob(self):
    return self._config.getfloat(self, 'embed_keep_prob')
  
#***************************************************************
class FormMultivocab(Multivocab, cv.FormVocab):
  _token_vocab_class = tv.FormTokenVocab
  _subtoken_vocab_class = sv.FormSubtokenVocab
  _pretrained_vocab_class = pv.FormPretrainedVocab
class LemmaMultivocab(Multivocab, cv.LemmaVocab):
  _token_vocab_class = tv.LemmaTokenVocab
  _subtoken_vocab_class = sv.LemmaSubtokenVocab
  _pretrained_vocab_class = pv.LemmaPretrainedVocab
class UPOSMultivocab(Multivocab, cv.UPOSVocab):
  _token_vocab_class = tv.UPOSTokenVocab
  _subtoken_vocab_class = sv.UPOSSubtokenVocab
  _pretrained_vocab_class = pv.UPOSPretrainedVocab
class XPOSMultivocab(Multivocab, cv.XPOSVocab):
  _token_vocab_class = tv.XPOSTokenVocab
  _subtoken_vocab_class = sv.XPOSSubtokenVocab
  _pretrained_vocab_class = pv.XPOSPretrainedVocab
class DeprelMultivocab(Multivocab, cv.DeprelVocab):
  _token_vocab_class = tv.DeprelTokenVocab
  _subtoken_vocab_class = sv.DeprelSubtokenVocab
  _pretrained_vocab_class = pv.DeprelPretrainedVocab
