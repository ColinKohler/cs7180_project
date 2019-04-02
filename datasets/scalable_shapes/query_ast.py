import numpy as np
import numpy.random as npr

#=============================================================================#
#                                 Base Types                                  #
#=============================================================================#

NULL_PROP = 0

RED = 1
GREEN = 2
BLUE = 3
COLOR_PROPERTY_STRS = {RED : 'red', GREEN : 'green', BLUE : 'blue'}
COLOR_PROPERTY_INTS = {'red': RED, 'green': GREEN, 'blue': BLUE}

ELLIPSE = 1
PLUS = 2
SHAPE_PROPERTY_STRS = {ELLIPSE : 'ellipse', PLUS : 'plus'}
SHAPE_PROPERTY_INTS = {'ellipse': ELLIPSE, 'plus': PLUS}

class Op(object):
  def eval(self, sample):
    return sample
  def query(self, parens=False):
    return ""
  def __call__(self, sample):
    self.eval(sample)

#=============================================================================#
#                               Nullary Ops                                   #
#=============================================================================#
# Have no inputs, but do have values.
# Examples:
#   x = Property([Color(RED),Shape(ELLIPSE)])
#   y = Color(RED)
#   z = Property([Color(RED)])
#

class NullaryOp(Op):
  def __init__(self, value = None):
    self.value = value
    
class Color(NullaryOp):
  def eval(self, sample):
    return sample[0] == self.value

  def query(self, single=True, parens=False):
    if single:
      return COLOR_PROPERTY_STRS[self.value] + ' shape'
    return COLOR_PROPERTY_STRS[self.value]

  def sample(self, depth, ops, prob):
    self.value= npr.choice(list(COLOR_PROPERTY_INTS.values()))

class Shape(NullaryOp):
  def eval(self, sample):
    return sample[1] == self.value

  def query(self, single=True, parens=False):
    return SHAPE_PROPERTY_STRS[self.value]

  def sample(self, depth, ops, prob):
    self.value=npr.choice(list(SHAPE_PROPERTY_INTS.values()))

class Property(NullaryOp):  #Value is assumed to be a list.
  def eval(self, sample):
    stack = np.stack([property.eval(sample) for property in self.value])
    return np.logical_and.reduce(stack)

  def query(self, parens=False):
    single = (len(self.value) == 1)
    return ' '.join([val.query(single) for val in self.value])

  def sample(self, depth, ops, prob):
    self.value=[Color(npr.choice(list(COLOR_PROPERTY_INTS.values()))),
                Shape(npr.choice(list(SHAPE_PROPERTY_INTS.values())))]

#=============================================================================#
#                                 Unary Ops                                   #
#=============================================================================#
# Has one input.

class UnaryOp(Op):
  def __init__(self, input = None):
    self.input = input
    
  def sample(self, depth, ops, prob):
    if depth == 0:
      self.input = generateRandomNullary()
    else:
      self.input = np.random.choice(ops,1,p=prob)[0]
      self.input.sample(depth-1, ops, prob)

class Is(UnaryOp):
  def eval(self, sample):
    return np.any(self.input.eval(sample))

  def query(self, parens=False):
    if parens:
      return '( is {} )'.format(self.input.query(parens=True))
    return 'is {}'.format(self.input.query(parens=False))

class Count(UnaryOp):
  def eval(self, sample):
    return np.sum(self.input.eval(sample))

  def query(self, parens=False):
    if parens:
      return '( count {} )'.format(self.input.query(parens=True))
    return 'count {}'.format(self.input.query(parens=False))

#=============================================================================#
#                                 Binary Ops                                  #
#=============================================================================#

class BinaryOp(Op):
  def __init__(self, left = None, right = None):
    self.left = left
    self.right = right

  def sample(self, depth, ops, prob):
    if depth == 0:
      self.left = generateRandomNullary()
      self.right = generateRandomNullary()
    else:
      self.left = np.random.choice(ops,1,p=prob)[0]
      self.left.sample(depth-1, ops, prob)
      self.right = np.random.choice(ops,1,p=prob)[0]
      self.right.sample(depth-1, ops, prob)

class And(BinaryOp):
  def eval(self, sample):
    return np.logical_and(self.left.eval(sample), self.right.eval(sample))

  def query(self, parens=False):
    if parens:
      return '( {} and {} )'.format(self.left.query(parens=True), self.right.query(parens=True))
    return '{} and {}'.format(self.left.query(parens=False), self.right.query(parens=False))

class Or(BinaryOp):
  def eval(self, sample):
    return np.logical_or(self.left.eval(sample), self.right.eval(sample))

  def query(self, parens=False):
    if parens:
      return '( {} or {} )'.format(self.left.query(parens=True), self.right.query(parens=True))
    return '{} or {}'.format(self.left.query(parens=False), self.right.query(parens=False))

def translateMask(mask,dx,dy):
  pad_mask = np.pad(mask, ((2,2),(2,2)), 'constant')
  push_mask = pad_mask[2-dy:-2-dy,2-dx:-2-dx]
  return push_mask

class RelDir(BinaryOp):
  def __init__(self, left, right, dirVec, name):
    BinaryOp.__init__(self,left,right)
    self.dirname = name
    self.dx, self.dy = dirVec
    self.axis = 1
    self.rev = lambda arr : arr
    if self.dy != 0:
      self.axis = 0
    if (self.dx < 0) or (self.dy < 0):
      self.rev = lambda arr : np.flip(arr, self.axis)

  def eval(self, sample):
    key = self.right.eval(sample)
    push_key = translateMask(key,self.dx,self.dy)
    push_key = self.rev(np.logical_or.accumulate(self.rev(push_key),self.axis))
    return np.logical_and(self.left.eval(sample), push_key)

  def query(self, parens=False):
    if parens:
      return '( {} {} {} )'.format(self.left.query(parens=True), self.dirname, self.right.query(parens=True))
    return '{} {} {}'.format(self.left.query(parens=False), self.dirname, self.right.query(parens=False))

class Above(RelDir):
  def __init__(self,left = None,right = None):
    RelDir.__init__(self,left,right,(0,-1),"above")

class Below(RelDir):
  def __init__(self,left = None,right = None):
    RelDir.__init__(self,left,right,(0,1),"below")

class Left(RelDir):
  def __init__(self,left = None,right = None):
    RelDir.__init__(self,left,right,(-1,0),"left of")

class Right(RelDir):
  def __init__(self,left = None,right = None):
    RelDir.__init__(self,left,right,(1,0),"right of")

class Near(BinaryOp):
  def eval(self, sample):
    key = self.right.eval(sample)
    moves = [(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)]
    stack = np.stack([translateMask(key,dx,dy) for dx,dy in moves])
    push_key = np.logical_or.reduce(stack)
    return np.logical_and(self.left.eval(sample), push_key)

  def query(self, parens=False):
    if parens:
      return '( {} near {} )'.format(self.left.query(parens=True), self.right.query(parens=True))
    return '{} near {}'.format(self.left.query(parens=False), self.right.query(parens=False))


#=============================================================================#
#                                 Boolean Ops                                  #
#=============================================================================#


class BoolBinaryOp(Op):
  def __init__(self, left = None, right = None):
    self.left = left
    self.right = right

class BoolAnd(BoolBinaryOp):
  def eval(self, sample):
    return (self.left.eval(sample) and self.right.eval(sample))

  def query(self, parens=False):
    if parens:
      return '( {} and {} )'.format(self.left.query(parens=True), self.right.query(parens=True))
    return '{} and {}'.format(self.left.query(parens=False), self.right.query(parens=False))

class BoolOr(BoolBinaryOp):
  def eval(self, sample):
    return (self.left.eval(sample) or self.right.eval(sample))
    
  def query(self, parens=False):
    if parens:
      return '( {} or {} )'.format(self.left.query(parens=True), self.right.query(parens=True))
    return '{} or {}'.format(self.left.query(parens=False), self.right.query(parens=False))


#=============================================================================#
#                                 Helpers                                     #
#=============================================================================#
def generateRandomProperty():
  #rand = npr.randint(2)
  if rand == 0:
    return Property([Color(npr.choice(list(COLOR_PROPERTY_INTS.values()))),
                     Shape(npr.choice(list(SHAPE_PROPERTY_INTS.values())))])
  #if rand == 0:
  #  return Property([Color(npr.choice(list(COLOR_PROPERTY_INTS.values())))])
  #else:
  #  return Property([Shape(npr.choice(list(SHAPE_PROPERTY_INTS.values())))])

def generateRandomNullary(property_prob = 0.33):
  rand = npr.randint(2)
  randn = npr.rand()
  # if rand == 0:
  #   return Property([Color(npr.choice(list(COLOR_PROPERTY_INTS.values()))),
  #                    Shape(npr.choice(list(SHAPE_PROPERTY_INTS.values())))])
  if randn < property_prob:
    return Property([Color(npr.choice(list(COLOR_PROPERTY_INTS.values()))),
                     Shape(npr.choice(list(SHAPE_PROPERTY_INTS.values())))])
  else:
    if rand == 0:
      return Color(npr.choice(list(COLOR_PROPERTY_INTS.values())))
    else:
      return Shape(npr.choice(list(SHAPE_PROPERTY_INTS.values())))
    
def generateRandomRelational(prop_1, prop_2):
  relational = npr.choice([Above, Below, Left, Right])
  return relational(prop_1, prop_2)
