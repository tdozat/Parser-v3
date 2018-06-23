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

#***************************************************************
class CoNLLUVocab():
  """"""
  
  _field = None
  _n_splits = None
  _conllu_idx = None
  
  #=============================================================
  @property
  def n_splits(self):
    return self._n_splits
  @property
  def field(self):
    return self._field
  @property
  def conllu_idx(self):
    return self._conllu_idx
  
#***************************************************************
class IDVocab(CoNLLUVocab):
  _field = 'id'
  _conllu_idx = 0
class FormVocab(CoNLLUVocab):
  _field = 'form'
  _conllu_idx = 1
class LemmaVocab(CoNLLUVocab):
  _field = 'lemma'
  _conllu_idx = 2
class UPOSVocab(CoNLLUVocab):
  _field = 'upos'
  _conllu_idx = 3
class XPOSVocab(CoNLLUVocab):
  _field = 'xpos'
  _conllu_idx = 4
class UFeatsVocab(CoNLLUVocab):
  _field = 'ufeats'
  _conllu_idx = 5
class DepheadVocab(CoNLLUVocab):
  _field = 'dephead'
  _conllu_idx = 6
class DeprelVocab(CoNLLUVocab):
  _field = 'deprel'
  _conllu_idx = 7
class SemheadVocab(CoNLLUVocab):
  _field = 'semhead'
  _conllu_idx = 8
class SemrelVocab(CoNLLUVocab):
  _field = 'semrel'
  _conllu_idx = 8
