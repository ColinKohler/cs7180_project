import numpy as np

#=============================================================================#
#                                 Base Types                                  #
#=============================================================================#

EMPTY = 0
RED_ELLIPSE = 1
GREEN_ELLIPSE = 2
BLUE_ELLIPSE = 3
RED_PLUS = 4
GREEN_PLUS =  5
BLUE_PLUS = 6

PROPERTY_QUERIES = {1 : 'red ellipse', 2 : 'green ellipse', 3 : 'blue ellipse',
                    4 : 'red plus',    5 : 'green plus',    6 : 'blue plus'}

class Property(object):
  def __init__(self, value):
    self.value = value

  def eval(self, sample):
    return sample == self.value

  def query(self):
    return PROPERTY_QUERIES[self.value]

#=============================================================================#
#                               Nullary Ops                                   #
#=============================================================================#

class NullaryOp(object):
  def __init__(self):
    pass

#=============================================================================#
#                                 Unary Ops                                   #
#=============================================================================#

class UnaryOp(object):
  def __init__(self, value):
    self.value = value

class Is(UnaryOp):
  def eval(self, sample):
    return np.any(self.value.eval(sample))

  def query(self):
    return 'Is {}'.format(self.value.query())

class Count(UnaryOp):
  def eval(self, sample):
    pass

  def query(self):
    return 'Count {}'.format(self.value.query())

#=============================================================================#
#                                 Binary Ops                                  #
#=============================================================================#

class BinaryOp(object):
  def __init__(self, left, right):
    self.left = left
    self.right = right

class And(BinaryOp):
  def eval(self, sample):
    return np.logical_and(self.left.eval(sample), self.right.eval(sample))

  def query(self):
    return '{} and {}'.format(self.left.query(), self.right.query())

class Or(BinaryOp):
  def eval(self, sample):
    return np.logical_or(self.left.eval(sample), self.right.eval(sample))

  def query(self):
    return '{} or {}'.format(self.left.query(), self.right.query())

class Above(BinaryOp):
  def eval(self, sample):
    key = self.right.eval(sample)
    push_key = np.vstack((key[1:], np.zeros((key.shape[1]))))
    push_key = np.logical_or.accumulate(push_key[::-1])[::-1]
    return np.logical_and(self.left.eval(sample), push_key)

  def query(self):
    return '{} above {}'.format(self.left.query(), self.right.query())

class Below(BinaryOp):
  def eval(self, sample):
    pass

  def query(self):
    return '{} below {}'.format(self.left.query(), self.right.query())

class Left(BinaryOp):
  def eval(self, sample):
    pass

  def query(self):
    return '{} left {}'.format(self.left.query(), self.right.query())

class Right(BinaryOp):
  def eval(self, sample):
    pass

  def query(self):
    return '{} right {}'.format(self.left.query(), self.right.query())

class Near(BinaryOp):
  def eval(self, sample):
    pass

  def query(self):
    return '{} near {}'.format(self.left.query(), self.right.query())

