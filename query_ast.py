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


#=============================================================================#
#                               Nullary Ops                                   #
#=============================================================================#
# Have no inputs, but do have values.

class NullaryOp(object):
  def __init__(self, value):
    self.value = value

class Property(NullaryOp):
  def eval(self, sample):
    return sample == self.value  

  def query(self):
    return PROPERTY_QUERIES[self.value]


#=============================================================================#
#                                 Unary Ops                                   #
#=============================================================================#
# Has one input.

class UnaryOp(object):
  def __init__(self, input):
    self.input = input
	
class Is(UnaryOp):
   def eval(self, sample):
    pass

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
    return np.logical_and(self.left.eval(), self.right.eval())

  def query(self):
    return '{} and {}'.format(self.left.query(), self.right.query())

class Or(BinaryOp):
  def eval(self, sample):
    return np.logical_or(self.left.eval(), self.right.eval())

  def query(self):
    return '{} or {}'.format(self.left.query(), self.right.query())

class Above(BinaryOp):
  def eval(self, sample):
    return

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

