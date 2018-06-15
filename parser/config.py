#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import glob

try:
  from ConfigParser import SafeConfigParser, NoOptionError
except ImportError:
  from configparser import SafeConfigParser, NoOptionError
  
#***************************************************************
# NOTE kwarg syntax should be 'SectionName'={'option1': value1, 'option2': value2}
class Config(SafeConfigParser, object):
  """"""
  
  #=============================================================
  def __init__(self, config_file='', **kwargs):
    """"""
    
    #super(Config, self).__init__(defaults=kwargs.pop('DEFAULT', {}))
    super(Config, self).__init__()
    self.read([os.path.join('config', 'defaults.cfg'), config_file])
    for section, options in six.iteritems(kwargs):
      if section != 'DEFAULT' and not self.has_section(section):
        self.add_section(section)
      for option, value in six.iteritems(options):
        self.set(section, option, str(value))
    return
  
  #=============================================================
  def _get_value(self, config_func, obj, option):
    """"""
    
    if '_'+option in obj.__dict__:
      return obj.__dict__['_'+option]
    cls = obj.__class__
    superclasses = [superclass.__name__ for superclass in cls.__mro__]
    for superclass in superclasses:
      if self.has_section(superclass) and \
         self.has_option(superclass, option):
        try:
          value = config_func(superclass, option)
        except ValueError:
          if self.get(superclass, option) == 'None':
            value = None
          else:
            raise
        break
    else:
      raise NoOptionError(option, superclasses)
    if value == 'None':
      value = None
    obj.__dict__['_'+option] = value
    return value
  
  #-------------------------------------------------------------
  def _get_list(self, lst):
    if lst is None:
      return lst
    lst = lst.split(':')
    i = 0
    while i < len(lst):
      if lst[i].endswith('\\'):
        lst[i] = ':'.join([lst[i].rstrip('\\'), lst.pop(i+1)])
      else:
        i += 1
    return lst
  
  #-------------------------------------------------------------
  def _glob_list(self, lst):
    if lst is None:
      return lst
    globbed = []
    for elt in lst:
      globs = glob.glob(elt)
      if len(globs) == 0:
        raise ValueError('Glob of %s yielded no files' % elt)
      globbed.extend(glob.glob(elt))
    return globbed
  
  #=============================================================
  def getstr(self, obj, option):
    return self._get_value(super(Config, self).get, obj, option)
  def getint(self, obj, option):
    return self._get_value(super(Config, self).getint, obj, option)
  def getfloat(self, obj, option):
    return self._get_value(super(Config, self).getfloat, obj, option)
  def getboolean(self, obj, option):
    return self._get_value(super(Config, self).getboolean, obj, option)
  def getlist(self, obj, option):
    if self._get_value(super(Config, self).get, obj, option) in ('', None):
      return []
    else:
      return self._get_list(self._get_value(super(Config, self).get, obj, option))
  def getfiles(self, obj, option):
    return self._glob_list(self.getlist(obj, option))
  
  #=============================================================
  def update(self, **kwargs):
    for section, options in six.iteritems(kwargs):
      for option, value in six.iteritems(options):
        self.set(section, option, str(value))
    return 
  def iteritems(self):
    for section in self.sections():
      yield section, {super(Config, self).get(section, option) for option in self.options(section)}
  def copy(self):
    config = Config()
    for section, options in six.iteritems(self):
      config.add_section(section)
      for option, value in six.iteritems(options):
        config.set(section, option, value)
    return config
  
  #=============================================================
  @property
  def save_dir(self):
    return self.getstr(self, 'save_dir')
