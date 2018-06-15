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
import re
import codecs
from collections import Counter

import numpy as np
import tensorflow as tf

from parser.structs.buckets import DictMultibucket
 
#***************************************************************
class CoNLLUDataset(set):
  """"""
  
  #=============================================================
  def __init__(self, conllu_files, vocabs, prefix_root=False, postfix_root=False, config=None):
    """"""
    
    super(CoNLLUDataset, self).__init__(vocabs)
    
    self._multibucket = DictMultibucket(vocabs, max_buckets=config.getint(self, 'max_buckets'), config=config)
    self._is_open = False
    self._config = config
    self._prefix_root = prefix_root
    self._postfix_root = postfix_root
    self._filenames = []
    
    self.add_files(conllu_files)
    return
  
  #=============================================================
  def add_files(self, conllu_files):
    """"""
    
    with self.open():
      for i, conllu_file in enumerate(conllu_files):
        file_index = i + len(self._filenames)
        for sent in self.itersents(conllu_file):
          self.add(sent, file_index)
      self._filenames.extend(conllu_files)
    return
  
  #=============================================================
  def reset(self, conllu_files):
    """"""
    
    self._multibucket.reset()
    for vocab in self:
      vocab.reset()
    
    self.add_files(conllu_files)
    return
  
  #=============================================================
  def open(self):
    """"""
    
    self._multibucket.open()
    for vocab in self:
      vocab.open()
    self._is_open = True
    return self
    
  #=============================================================
  def add(self, sent, file_index):
    """"""
    
    assert self._is_open, 'The CoNLLUDataset is not open for adding entries'
    
    sent_tokens = {}
    sent_indices = {}
    for vocab in self:
      try:
        tokens = [line[vocab.conllu_idx] for line in sent]
        if self.prefix_root:
          tokens.insert(0, vocab.get_root())
        if self.postfix_root:
          tokens.append(vocab.get_root())
      except:
        raise
      indices = vocab.add_sequence(tokens) # for graphs, list of (head, label) pairs
      sent_tokens[vocab.classname] = tokens
      sent_indices[vocab.classname] = indices
    self._multibucket.add(sent_indices, sent_tokens, file_index=file_index, length=len(sent)+1)
    return
  
  #=============================================================
  def close(self):
    """"""
    
    self._multibucket.close()
    for vocab in self:
      vocab.close()
    self._is_open = False
    return 
  
  #=============================================================
  def shuffled_batch_iterator(self):
    """"""
    
    assert self.batch_size > 0, 'batch_size must be > 0'
    
    batches = []
    bucket_indices = self._multibucket.bucket_indices
    for i in np.unique(bucket_indices):
      subdata = np.where(bucket_indices == i)[0]
      if len(subdata) > 0:
        np.random.shuffle(subdata)
        n_splits = max(subdata.shape[0] * self._multibucket.max_lengths[i] // self.batch_size, 1)
        batches.extend(np.array_split(subdata, n_splits))
    np.random.shuffle(batches)
    return iter(batches)
  
  #=============================================================
  def file_batch_iterator(self, file_index):
    """"""
    
    batch_size = self.batch_size
    assert batch_size > 0, 'batch_size must be > 0'
    
    file_indices = self._multibucket.file_indices
    bucket_indices = self._multibucket.bucket_indices
    
    file_indices = self._multibucket.file_indices
    file_i_indices = np.where(file_indices == file_index)[0]
    file_i_bucket_indices = bucket_indices[file_i_indices]
    for j in np.unique(file_i_bucket_indices):
      bucket_j_indices = np.where(file_i_bucket_indices == j)[0]
      file_i_bucket_j_indices = file_i_indices[bucket_j_indices]
      indices = np.array_split(file_i_bucket_j_indices, np.ceil(len(file_i_bucket_j_indices) * self._multibucket.max_lengths[j] / batch_size))
      for batch in indices:
        yield batch
    
  #=============================================================
  def set_placeholders(self, indices, feed_dict={}):
    """"""
    
    for vocab in self:
      data = self._multibucket.get_data(vocab.classname, indices)
      feed_dict = vocab.set_placeholders(data, feed_dict=feed_dict)
    return feed_dict
  
  ##=============================================================
  #def preds_to_toks(self, predictions, field2vocab):
  #  """"""
  #  
  #  tokens = {}
  #  
  #  for vocab_classname, preds in six.iteritems(predictions):
  #    try:
  #      tokens[vocab_classname] = self[vocab_classname][preds]
  #    except:
  #      raise 
  #  return tokens
  
  #=============================================================
  def get_tokens(self, indices):
    """"""
    
    token_dict = {}
    for vocab in self:
      token_dict[vocab.field] = self._multibucket.get_tokens(vocab.classname, indices)
    return token_dict
  
  #=============================================================
  @staticmethod
  def itersents(conllu_file):
    """"""
    
    with codecs.open(conllu_file, encoding='utf-8', errors='ignore') as f:
      buff = []
      for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
          if not re.match('[0-9]+[-.][0-9]+', line):
            buff.append(line.split('\t'))
        elif buff:
          yield buff
          buff = []
      yield buff
  
  #=============================================================
  @property
  def n_sents(self):
    return len(self._lengths)
  @property
  def prefix_root(self):
    return self._prefix_root
  @property
  def postfix_root(self):
    return self._postfix_root
  @property
  def save_dir(self):
    return self._config.getstr(self, 'save_dir')
  @property
  def filenames(self):
    return list(self._filenames)
  @property
  def max_buckets(self):
    return self._config.getint(self, 'max_buckets')
  @property
  def batch_size(self):
    return self._config.getint(self, 'batch_size')
  @property
  def classname(self):
    return self.__class__.__name__
  
  #=============================================================
  def __enter__(self):
    return self
  def __exit__(self, exception_type, exception_value, trace):
    if exception_type is not None:
      raise
    self.close()
    return

#***************************************************************
class CoNLLUTrainset(CoNLLUDataset):
  def __init__(self, *args, config=None, **kwargs):
    super(CoNLLUTrainset, self).__init__(config.getfiles(self, 'train_conllus'), *args, config=config, **kwargs)

class CoNLLUDevset(CoNLLUDataset):
  def __init__(self, *args, config=None, **kwargs):
    super(CoNLLUDevset, self).__init__(config.getfiles(self, 'dev_conllus'), *args, config=config, **kwargs)

class CoNLLUTestset(CoNLLUDataset):
  def __init__(self, *args, config=None, **kwargs):
    super(CoNLLUTestset, self).__init__(config.getfiles(self, 'test_conllus'), *args, config=config, **kwargs)
