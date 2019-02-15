import numpy as np

#=============================================================================#
#                                 Base Types                                  #
#=============================================================================#

NULL_PROP = 0

RED = 1
GREEN = 2
BLUE = 3
COLOR_PROPERTY_STRS = {1 : 'red', 2 : 'green', 3 : 'blue'}

ELLIPSE = 1
PLUS = 2
SHAPE_PROPERTY_STRS = {1 : 'ellipse', 2 : 'plus'}

#=============================================================================#
#                               Nullary Ops                                   #
#=============================================================================#
# Have no inputs, but do have values.

class NullaryOp(object):
  def __init__(self, value):
    self.value = value

class Color(NullaryOp):
  def eval(self, sample):
    return sample == self.value

  def query(self):
    return COLOR_PROPERTY_STRS[self.value]

class Shape(NullaryOp):
  def eval(self, sample):
    return sample == self.value

  def query(self):
    return SHAPE_PROPERTY_STRS[self.value]

class Property(NullaryOp):
  def eval(self, sample):
    return np.logical_and(self.value[0].eval(sample[0]),
                          self.value[1].eval(sample[1]))

  def query(self):
    return '{} {}'.format(self.value[0].query(), self.value[1].query())

#=============================================================================#
#                                 Unary Ops                                   #
#=============================================================================#
# Has one input.

class UnaryOp(object):
  def __init__(self, input):
    self.input = input

class Is(UnaryOp):
  def eval(self, sample):
    return np.any(self.input.eval(sample))

  def query(self):
    return 'Is {}'.format(self.input.query())

class Count(UnaryOp):
  def eval(self, sample):
    pass

  def query(self):
    return 'Count {}'.format(self.input.query())

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

