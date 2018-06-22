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
try:
  import cPickle as pkl
except ImportError:
  import pickle as pkl
  
import curses
import time

import numpy as np
import tensorflow as tf

from parser.neural import nn
from scripts.chuliu_edmonds import chuliu_edmonds_one_root

#***************************************************************
class GraphOutputs(object):
  """"""
  
  _dataset = None
  
  _print_mapping = [('form', 'Form'),
                    ('lemma', 'Lemma'),
                    ('upos', 'UPOS'),
                    ('xpos', 'XPOS'),
                    ('dephead', 'UAS'),
                    ('deprel', 'OLS'),
                    ('deptree', 'LAS'),
                    ('semhead', 'UF1'),
                    ('semrel', 'OLS'),
                    ('semgraph', 'LF1')]
  
  #=============================================================
  def __init__(self, outputs, tokens, load=False, evals=None, factored_deptree=None, factored_semgraph=None, config=None):
    """"""
    
    self._factored_deptree = factored_deptree
    self._factored_semgraph = factored_semgraph
    self._config = config
    self._evals = evals or list(outputs.keys())
    #self._evals = config.getlist(self, 'evals')
    valid_evals = set([print_map[0] for print_map in self._print_mapping])
    
    for eval_ in list(self._evals):
      assert eval_ in valid_evals
    self._loss = tf.add_n([tf.where(tf.is_finite(output['loss']), output['loss'], 0.) for output in outputs.values()])
    self._accuracies = {'total': tokens}
    self._probabilities = {}
    self.time = None
    
    #-----------------------------------------------------------
    for field in outputs:
      self._probabilities[field] = outputs[field].pop('probabilities')
      self._accuracies[field] = outputs[field]
    
    #-----------------------------------------------------------
    filename = os.path.join(self.save_dir, '{}.pkl'.format(self.dataset))
    # TODO make a separate History object
    if load and os.path.exists(filename):
      with open(filename, 'rb') as f:
        self.history = pkl.load(f)
    else:
      self.history = {
        'total': {'n_batches' : 0,
                  'n_tokens': 0,
                  'n_sequences': 0,
                  'total_time': 0},
        'speed': {'toks/sec': [],
                  'seqs/sec': [],
                  'bats/sec': []}
        }
      for field in self._accuracies:
        if field == 'semgraph':
          for string in ('head', 'graph'):
            self.history['sem'+string] = {
              'loss': [0],
              'tokens': [0],
              'fp_tokens': 0,
              'fn_tokens': 0,
              'sequences': [0]
            }
          if self._factored_semgraph:
            self.history['semrel'] = {
              'loss': [0],
              'tokens': [0],
              'n_edges': 0,
              'sequences': [0]
            }
            
        elif field == 'deptree':
          for string in ('head', 'tree'):
            self.history['dep'+string] = {
              'loss': [0],
              'tokens': [0],
              'sequences': [0]
            }
          if self._factored_deptree:
            self.history['deprel'] = {
              'loss': [0],
              'tokens': [0],
              'sequences': [0]
            }
        elif field not in ('speed', 'total'):
          self.history[field] ={
            'loss': [0],
            'tokens': [0],
            'sequences': [0]
          }
    self.predictions = {'indices': []}
    return
  
  #=============================================================
  def probs_to_preds(self, probabilities, lengths):
    """"""
    
    predictions = {}
    
    if 'form' in probabilities:
      form_probs = probabilities['form']
      if isinstance(form_probs, tuple):
        form_samples, form_probs = form_probs
        form_preds = np.argmax(form_probs, axis=-1)
        predictions['form'] = form_samples[np.arange(len(form_preds)), form_preds]
      else:
        form_preds = np.argmax(form_probs, axis=-1)
        predictions['form'] = form_preds
    if 'lemma' in probabilities:
      lemma_probs = probabilities['lemma']
      lemma_preds = np.argmax(lemma_probs, axis=-1)
      predictions['lemma'] = lemma_preds
    if 'upos' in probabilities:
      upos_probs = probabilities['upos']
      #print(upos_probs[0,1])
      #input()
      upos_preds = np.argmax(upos_probs, axis=-1)
      predictions['upos'] = upos_preds
    if 'xpos' in probabilities:
      xpos_probs = probabilities['xpos']
      xpos_preds = np.argmax(xpos_probs, axis=-1)
      predictions['xpos'] = xpos_preds
    #if 'head' in probabilities: # TODO MST algorithms
    #  head_probs = probabilities['head']
    #  head_preds = np.argmax(head_probs, axis=-1)
    #  predictions['head'] = head_preds
    if 'deptree' in probabilities:
      # (n x m x m x c)
      deptree_probs = probabilities['deptree']
      if self._factored_deptree:
        # (n x m x m x c) -> (n x m x m)
        dephead_probs = deptree_probs.sum(axis=-1)
        # (n x m x m) -> (n x m)
        #dephead_preds = np.argmax(dephead_probs, axis=-1)
        try:
          dephead_preds = np.zeros(dephead_probs.shape[:2], dtype=np.int32)
          for i, (_dephead_probs, length) in enumerate(zip(dephead_probs, lengths)):
            #print(_dephead_probs)
            #input()
            cle = chuliu_edmonds_one_root(_dephead_probs[:length, :length])
            dephead_preds[i, :length] = cle
        except:
          with open('debug.log', 'w') as f:
            f.write('{}\n'.format(cle))
          raise
        # ()
        bucket_size = dephead_preds.shape[1]
        # (n x m) -> (n x m x m)
        one_hot_dephead_preds = (np.arange(bucket_size) == dephead_preds[...,None]).astype(int)
        # (n x m x m) * (n x m x m x c) -> (n x m x c)
        deprel_probs = np.einsum('ijk,ijkl->ijl', one_hot_dephead_preds, deptree_probs)
        # (n x m x c) -> (n x m)
        deprel_preds = np.argmax(deprel_probs, axis=-1)
      else:
        # (), ()
        bucket_size, n_classes = deptree_probs.shape[-2:]
        # (n x m x m x c) -> (n x m x mc)
        deptree_probs = deptree_probs.reshape([-1, bucket_size, bucket_size*n_classes])
        # (n x m x mc) -> (n x m)
        deptree_preds = np.argmax(deptree_probs, axis=-1)
        # (n x m) -> (n x m)
        dephead_preds = deptree_preds // bucket_size
        deprel_preds = deptree_preds % n_classes
      predictions['dephead'] = dephead_preds
      predictions['deprel'] = deprel_preds
    if 'semgraph' in probabilities:
      # (n x m x m x c)
      semgraph_probs = probabilities['semgraph']
      if self._factored_semgraph:
        # (n x m x m x c) -> (n x m x m)
        semhead_probs = semgraph_probs.sum(axis=-1)
        # (n x m x m) -> (n x m x m)
        semhead_preds = np.where(semhead_probs >= .5, 1, 0)
        # (n x m x m x c) -> (n x m x m)
        semrel_preds = np.argmax(semgraph_probs, axis=-1)
        # (n x m x m) (*) (n x m x m) -> (n x m x m)
        semgraph_preds = semhead_preds * semrel_preds
      else:
        # (n x m x m x c) -> (n x m x m)
        semgraph_preds = np.argmax(semgraph_probs, axis=-1)
      predictions['semrel'] = sparse_semgraph_preds = []
      for i in xrange(len(semgraph_preds)):
        sparse_semgraph_preds.append([])
        for j in xrange(len(semgraph_preds[i])):
          sparse_semgraph_preds[-1].append([])
          for k, pred in enumerate(semgraph_preds[i,j]):
            if pred:
              sparse_semgraph_preds[-1][-1].append((k, semgraph_preds[i,j,k]))
    return predictions
  
  #=============================================================
  def cache_predictions(self, tokens, indices):
    """"""
    
    self.predictions['indices'].extend(indices)
    for field in tokens:
      if field not in self.predictions:
        self.predictions[field] = []
      self.predictions[field].extend(tokens[field])
    return
  
  #=============================================================
  def print_current_predictions(self):
    """"""
    
    order = np.argsort(self.predictions['indices'])
    fields = ['form', 'lemma', 'upos', 'xpos', 'morph', 'dephead', 'deprel', 'semrel', 'misc']
    for i in order:
      j = 1
      token = []
      while j < len(self.predictions['id'][i]):
        token = [self.predictions['id'][i][j]]
        for field in fields:
          if field in self.predictions:
            token.append(self.predictions[field][i][j])
          else:
            token.append('_')
        print('\t'.join(token))
        j += 1
      print('')
    self.predictions = {'indices': []}
    return

  #=============================================================
  def dump_current_predictions(self, f):
    """"""
    
    order = np.argsort(self.predictions['indices'])
    fields = ['form', 'lemma', 'upos', 'xpos', 'morph', 'dephead', 'deprel', 'semrel', 'misc']
    for i in order:
      j = 1
      token = []
      while j < len(self.predictions['id'][i]):
        token = [self.predictions['id'][i][j]]
        for field in fields:
          if field in self.predictions:
            token.append(self.predictions[field][i][j])
          else:
            token.append('_')
        f.write('\t'.join(token)+'\n')
        j += 1
      f.write('\n')
    self.predictions = {'indices': []}
    return
  
  #=============================================================
  def compute_token_accuracy(self, field):
    """"""
    
    return self.history[field]['tokens'][-1] / (self.history['total']['n_tokens'] + 1e-12)
  
  def compute_token_F1(self, field):
    """"""
    
    precision = self.history[field]['tokens'][-1] / (self.history[field]['tokens'][-1] + self.history[field]['fp_tokens'] + 1e-12)
    recall = self.history[field]['tokens'][-1] / (self.history[field]['tokens'][-1] + self.history[field]['fn_tokens'] + 1e-12)
    return 2 * (precision * recall) / (precision + recall + 1e-12)
  
  def compute_sequence_accuracy(self, field):
    """"""
    
    return self.history[field]['sequences'][-1] / self.history['total']['n_sequences']
  
  #=============================================================
  def get_current_accuracy(self):
    """"""
    
    token_accuracy = 0
    for field in self.history:
      if field in self.evals:
        if field.startswith('sem'):
          token_accuracy += np.log(self.compute_token_F1(field)+1e-12)
        else:
          token_accuracy += np.log(self.compute_token_accuracy(field)+1e-12)
    token_accuracy /= len(self.evals)
    return np.exp(token_accuracy) * 100
  
  #=============================================================
  def get_current_geometric_accuracy(self):
    """"""
    
    token_accuracy = 0
    for field in self.history:
      if field in self.evals:
        if field.startswith('sem'):
          token_accuracy += np.log(self.compute_token_F1(field)+1e-12)
        else:
          token_accuracy += np.log(self.compute_token_accuracy(field)+1e-12)
    token_accuracy /= len(self.evals)
    return np.exp(token_accuracy) * 100
  
  #=============================================================
  def restart_timer(self):
    """"""
    
    self.time = time.time()
    return
  
  #=============================================================
  def update_history(self, outputs):
    """"""
    
    self.history['total']['total_time'] += time.time() - self.time
    self.time = None
    self.history['total']['n_batches'] += 1
    self.history['total']['n_tokens'] += outputs['total']['n_tokens']
    self.history['total']['n_sequences'] += outputs['total']['n_sequences']
    for field, output in six.iteritems(outputs):
      if field == 'semgraph':
        if self._factored_semgraph:
          self.history['semrel']['loss'][-1] += output['label_loss']
          self.history['semrel']['tokens'][-1] += output['n_correct_label_tokens']
          self.history['semrel']['n_edges'] += output['n_true_positives'] + output['n_false_negatives']
          self.history['semrel']['sequences'][-1] += output['n_correct_label_sequences']
        self.history['semhead']['loss'][-1] += output['unlabeled_loss']
        self.history['semhead']['tokens'][-1] += output['n_unlabeled_true_positives']
        self.history['semhead']['fp_tokens'] += output['n_unlabeled_false_positives']
        self.history['semhead']['fn_tokens'] += output['n_unlabeled_false_negatives']
        self.history['semhead']['sequences'][-1] += output['n_correct_unlabeled_sequences']
        self.history['semgraph']['loss'][-1] += output['loss']
        self.history['semgraph']['tokens'][-1] += output['n_true_positives']
        self.history['semgraph']['fp_tokens'] += output['n_false_positives']
        self.history['semgraph']['fn_tokens'] += output['n_false_negatives']
        self.history['semgraph']['sequences'][-1] += output['n_correct_sequences']
      elif field == 'deptree':
        if self._factored_deptree:
          self.history['deprel']['loss'][-1] += output['label_loss']
          self.history['deprel']['tokens'][-1] += output['n_correct_label_tokens']
          self.history['deprel']['sequences'][-1] += output['n_correct_label_sequences']
        self.history['dephead']['loss'][-1] += output['unlabeled_loss']
        self.history['dephead']['tokens'][-1] += output['n_correct_unlabeled_tokens']
        self.history['dephead']['sequences'][-1] += output['n_correct_unlabeled_sequences']
        self.history['deptree']['loss'][-1] += output['loss']
        self.history['deptree']['tokens'][-1] += output['n_correct_tokens']
        self.history['deptree']['sequences'][-1] += output['n_correct_sequences']
      elif field != 'total':
        self.history[field]['loss'][-1] += output['loss']
        self.history[field]['tokens'][-1] += output['n_correct_tokens']
        self.history[field]['sequences'][-1] += output['n_correct_sequences']
    return
  
  #=============================================================
  def print_recent_history(self, stdscr=None):
    """"""
    
    n_batches = self.history['total']['n_batches']
    n_tokens = self.history['total']['n_tokens']
    n_sequences = self.history['total']['n_sequences']
    total_time = self.history['total']['total_time']
    self.history['total']['n_batches'] = 0
    self.history['total']['n_tokens'] = 0
    self.history['total']['n_sequences'] = 0
    self.history['total']['total_time'] = 0
    
    #-----------------------------------------------------------
    if stdscr is not None:
      stdscr.addstr('{:5}\n'.format(self.dataset.title()), curses.color_pair(1) | curses.A_BOLD)
      stdscr.clrtoeol()
    else:
      print('{:5}\n'.format(self.dataset.title()), end='')
    
    for field, string in self._print_mapping:
      if field in self.history:
        tokens = self.history[field]['tokens'][-1]
        if field in ('semgraph', 'semhead'):
          tp = self.history[field]['tokens'][-1]
          self.history[field]['tokens'][-1] = self.compute_token_F1(field) * 100
        elif field == 'semrel':
          n_edges = self.history[field]['n_edges']
          self.history[field]['tokens'][-1] *= 100 / n_edges
          self.history[field]['n_edges'] = 0
        else:
          self.history[field]['tokens'][-1] *= 100 / n_tokens
        self.history[field]['loss'][-1] /= n_batches
        self.history[field]['sequences'][-1] *= 100 / n_sequences
        loss = self.history[field]['loss'][-1]
        acc = self.history[field]['tokens'][-1]
        acc_seq = self.history[field]['sequences'][-1]
        if stdscr is not None:
          stdscr.addstr('{:5}'.format(string), curses.color_pair(6) | curses.A_BOLD)
          stdscr.addstr(' | ')
          stdscr.addstr('Loss: {:.2e}'.format(loss), curses.color_pair(3) | curses.A_BOLD)
          stdscr.addstr(' | ')
          stdscr.addstr('Acc: {:5.2f}'.format(acc), curses.color_pair(4) | curses.A_BOLD)
          stdscr.addstr(' | ')
          stdscr.addstr('Seq: {:5.2f}\n'.format(acc_seq), curses.color_pair(4) | curses.A_BOLD)
          stdscr.clrtoeol()
        else:
          print('{:5}'.format(string), end='')
          print(' | ', end='')
          print('Loss: {:.2e}'.format(loss), end='')
          print(' | ', end='')
          print('Acc: {:5.2f}'.format(acc), end='')
          print(' | ', end='')
          print('Seq: {:5.2f}\n'.format(acc_seq), end='')
        for key, value in six.iteritems(self.history[field]):
          if hasattr(value, 'append'):
            value.append(0)
          else:
            self.history[field][key] = 0
    
    self.history['speed']['toks/sec'].append(n_tokens / total_time)
    self.history['speed']['seqs/sec'].append(n_sequences / total_time)
    self.history['speed']['bats/sec'].append(n_batches / total_time)
    tps = self.history['speed']['toks/sec'][-1]
    sps = self.history['speed']['seqs/sec'][-1]
    bps = self.history['speed']['bats/sec'][-1]
    if stdscr is not None:
      stdscr.clrtoeol()
      stdscr.addstr('Speed', curses.color_pair(6) | curses.A_BOLD)
      stdscr.addstr(' | ')
      stdscr.addstr('Seqs/sec: {:6.1f}'.format(sps), curses.color_pair(5) | curses.A_BOLD)
      stdscr.addstr(' | ')
      stdscr.addstr('Bats/sec: {:4.2f}\n'.format(bps), curses.color_pair(5) | curses.A_BOLD)
      stdscr.clrtoeol()
      stdscr.addstr('Count', curses.color_pair(6) | curses.A_BOLD)
      stdscr.addstr(' | ')
      stdscr.addstr('Toks: {:6d}'.format(n_tokens), curses.color_pair(7) | curses.A_BOLD)
      stdscr.addstr(' | ')
      stdscr.addstr('Seqs: {:5d}\n'.format(n_sequences), curses.color_pair(7) | curses.A_BOLD)
    else:
      print('Speed', end='')
      print(' | ', end='')
      print('Seqs/sec: {:6.1f}'.format(sps), end='')
      print(' | ', end='')
      print('Bats/sec: {:4.2f}\n'.format(bps), end='')
      print('Count', end='')
      print(' | ', end='')
      print('Toks: {:6d}'.format(n_tokens), end='')
      print(' | ', end='')
      print('Seqs: {:5d}\n'.format(n_sequences), end='')
    filename = os.path.join(self.save_dir, '{}.pkl'.format(self.dataset))
    with open(filename, 'wb') as f:
      pkl.dump(self.history, f, protocol=pkl.HIGHEST_PROTOCOL)
    return
  
  #=============================================================
  @property
  def evals(self):
    return self._evals
  @property
  def accuracies(self):
    return dict(self._accuracies)
  @property
  def probabilities(self):
    return dict(self._probabilities)
  @property
  def loss(self):
    return self._loss
  @property
  def save_dir(self):
    return self._config.getstr(self, 'save_dir')
  @property
  def dataset(self):
    return self._dataset

#***************************************************************
class TrainOutputs(GraphOutputs):
  _dataset = 'train'
class DevOutputs(GraphOutputs):
  _dataset = 'dev'
