import time

#***************************************************************
class Timer(object):
  """"""
  
  #=============================================================
  def __init__(self, name):
    self._name = name
    self._start = None
    
  def __enter__(self):
    self._start = time.time()
    return self
  
  def __exit__(self, *args):
    print('{}: {:0.2f}'.format(self.name, time.time() - self.start))
    return

  #=============================================================
  @property
  def name(self):
    return self._name
  @property
  def start(self):
    return self._start
