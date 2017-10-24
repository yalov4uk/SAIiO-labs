#!/usr/bin/python2.7
from numpy import *

class ResourceAllocationSolver(object):
  def __init__(self, f):
    self.f = f
    self.n, self.c = f.shape
    self.B = zeros([self.n, self.c])
    self.B[0] = self.f[0]
    self.parent = zeros([self.n, self.c])
    self.parent[0] = range(self.c)
    self.x = []

  def generate_x(self, n, c):
    if (n > 0):
      self.generate_x(n - 1, c - self.parent[n][c])
    self.x.append(self.parent[n][c])

  def solve(self):
    for k in xrange(1, self.n):
      for y in xrange(self.c): 
        maximum = -inf
        for z in xrange(y + 1):
          curr = self.f[k][z] + self.B[k - 1][y-z]
          if curr > maximum:
            maximum = curr
            self.parent[k][y] = z
        self.B[k][y] = maximum
    self.generate_x(self.n - 1, self.c - 1)
    return self

if __name__ == '__main__':
  f = array([
    array([0, 3, 4, 5, 8, 9, 10]),
    array([0, 2, 3, 7, 9, 12, 13]),
    array([0, 1, 2, 6, 11, 11 ,13])
  ])
  obj = ResourceAllocationSolver(f).solve()
  print obj.B
  print obj.parent
  print obj.x