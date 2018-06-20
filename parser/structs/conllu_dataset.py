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
import zipfile
import gzip
try:
  import lzma
except ImportError:
  try:
    from backports import lzma
  except ImportError:
    import warnings
    warnings.warn('Install backports.lzma for xz support')
from collections import Counter

import numpy as np
import tensorflow as tf

from parser.structs.buckets import DictMultibucket
 
#***************************************************************
class CoNLLUDataset(set):
  """"""
  
  #=============================================================
  def __init__(self, conllu_files, vocabs, config=None):
    """"""
    
    super(CoNLLUDataset, self).__init__(vocabs)
    
    self._multibucket = DictMultibucket(vocabs, max_buckets=config.getint(self, 'max_buckets'), config=config)
    self._is_open = False
    self._config = config
    self._conllu_files = conllu_files
    assert len(conllu_files) > 0, "You didn't pass in any valid CoNLLU files! Maybe you got the path wrong?"
    self._cur_file_idx = -1
    
    self.load_next()
    return
  
  #=============================================================
  def reset(self):
    """"""
    
    self._multibucket.reset(self)
    for vocab in self:
      vocab.reset()
    return
    
  #=============================================================
  def load_next(self, file_idx=None):
    """"""
    
    if self._cur_file_idx == -1 or len(self.conllu_files) > 1:
      self.reset()
    
      if file_idx is None:
        self._cur_file_idx = (self._cur_file_idx + 1) % len(self.conllu_files)
        file_idx = self._cur_file_idx
      
      with self.open():
        for sent in self.itersents(self.conllu_files[file_idx]):
          self.add(sent)
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
  def add(self, sent):
    """"""
    
    assert self._is_open, 'The CoNLLUDataset is not open for adding entries'
    
    sent_tokens = {}
    sent_indices = {}
    for vocab in self:
      tokens = [line[vocab.conllu_idx] for line in sent]
      tokens.insert(0, vocab.get_root())
      indices = vocab.add_sequence(tokens) # for graphs, list of (head, label) pairs
      sent_tokens[vocab.classname] = tokens
      sent_indices[vocab.classname] = indices
    self._multibucket.add(sent_indices, sent_tokens, length=len(sent)+1)
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
  def batch_iterator(self, shuffle=False):
    """"""
    
    assert self.batch_size > 0, 'batch_size must be > 0'
    
    batches = []
    bucket_indices = self._multibucket.bucket_indices
    for i in np.unique(bucket_indices):
      subdata = np.where(bucket_indices == i)[0]
      if len(subdata) > 0:
        if shuffle:
          np.random.shuffle(subdata)
        n_splits = max(subdata.shape[0] * self._multibucket.max_lengths[i] // self.batch_size, 1)
        batches.extend(np.array_split(subdata, n_splits))
    if shuffle:
      np.random.shuffle(batches)
    return iter(batches)
  
  ##=============================================================
  #def file_batch_iterator(self, file_index):
  #  """"""
  #  
  #  batch_size = self.batch_size
  #  assert batch_size > 0, 'batch_size must be > 0'
  #  
  #  bucket_indices = self._multibucket.bucket_indices
  #  for i in np.unique(bucket_indices):
  #    subdata = np.where(bucket_indices == i)[0]
  #    if len(subdata) > 0:
  #      n_splits = max(subdata.shape[0] * self._multibucket.max_lengths[i] // self.batch_size, 1)
  #      batches.extend(np.array_split(subdata, n_splits))
  #  return iter(batches)
  #  
  #  #file_indices = self._multibucket.file_indices
  #  #file_i_indices = np.where(file_indices == file_index)[0]
  #  #file_i_bucket_indices = bucket_indices[file_i_indices]
  #  #for j in np.unique(file_i_bucket_indices):
  #  #  bucket_j_indices = np.where(file_i_bucket_indices == j)[0]
  #  #  file_i_bucket_j_indices = file_i_indices[bucket_j_indices]
  #  #  indices = np.array_split(file_i_bucket_j_indices, np.ceil(len(file_i_bucket_j_indices) * self._multibucket.max_lengths[j] / batch_size))
  #  #  for batch in indices:
  #  #    yield batch
    
  #=============================================================
  def set_placeholders(self, indices, feed_dict={}):
    """"""
    
    for vocab in self:
      data = self._multibucket.get_data(vocab.classname, indices)
      feed_dict = vocab.set_placeholders(data, feed_dict=feed_dict)
    return feed_dict
  
  #=============================================================
  def get_tokens(self, indices):
    """"""
    
    
    token_dict = {}
    for vocab in self:
      token_dict[vocab.field] = self._multibucket.get_tokens(vocab.classname, indices)
    lengths = self._multibucket.lengths[indices]
    return token_dict, lengths
  
  #=============================================================
  @staticmethod
  def itersents(conllu_file):
    """"""
    
    if conllu_file.endswith('.zip'):
      open_func = zipfile.Zipfile
      kwargs = {}
    elif conllu_file.endswith('.gz'):
      open_func = gzip.open
      kwargs = {}
    elif conllu_file.endswith('.xz'):
      open_func = lzma.open
      kwargs = {'errors': 'ignore'}
    else:
      open_func = codecs.open
      kwargs = {'errors': 'ignore'}
    
    with open_func(conllu_file, 'rb') as f:
      reader = codecs.getreader('utf-8')(f, **kwargs)
      buff = []
      for line in reader:
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
  def save_dir(self):
    return self._config.getstr(self, 'save_dir')
  @property
  def conllu_files(self):
    return list(self._conllu_files)
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
