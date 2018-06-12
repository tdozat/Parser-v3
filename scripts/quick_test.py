import codecs
from collections import Counter

def test(filename):
  duplicates = set()
  sent_id = None
  with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
    graph = Counter()
    for line in f:
      line = line.strip()
      if line.startswith('#'):
        sent_id = line
        line = ''
      if line:
        line = line.split('\t')
        semnodes = line[8]
        if semnodes != '_':
          graph.update(semnodes.split('|'))
      else:
        for semnode, count in graph.most_common():
          if count > 1:
            semrel = semnode.split(':', 1)[1]
            duplicates.add(semrel)
            if sent_id is not None:
              print(semrel, sent_id)
              input()
        graph = Counter()
  return duplicates

if __name__ == '__main__':
  import sys
  print(test(sys.argv[1]))
