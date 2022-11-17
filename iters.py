# some extra itertool stuff
# mutable repeat

from collections.abc import Iterator, Iterable
from itertools import repeat

repeat = repeat

def MutRep(val):
  while True:
    tmp = yield val
    if isinstance(tmp, tuple):
      [val] = tmp

def chain(*itrs):
  for itr in map(iter, itrs):
    tmp = yield from itr
    if tmp is not None: return tmp

def doFirst(f, itr):
  itr = iter(itr)
  yield f(next(itr))
  return (yield from itr)

def doLast(f, itr):
  itr = iter(itr)
  tmp = next(itr)
  while True:
      try:
        tmp2 = next(itr)
        yield tmp
        tmp = tmp2
      except StopIteration:
        yield f(tmp)
        break

def doWhile(s, f):
  while True:
    tmp = f()
    if not s(tmp): return
    yield tmp

def flatten(itr):
  for i in itr:
    if isinstance(i, Iterator | Iterable):
      yield from flatten(i)
    else:
      yield i

def doWhen(s, f, itr):
  for x in itr:
    if s(x): yield f(x)
    else: yield x

def noArg(itr):
  for i in itr: yield i()

def setret(val):
  def inner(): return val
  return inner

def chunk(itr, size):
    assert size > 0
    cur = []
    for i in itr:
        cur.append(i)
        if len(cur) == size:
            yield cur
            cur = []
    if cur: yield cur