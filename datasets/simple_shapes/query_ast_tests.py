import unittest
import numpy as np

import query_ast as q_ast
from query_ast import NULL_PROP
from query_ast import RED, GREEN, BLUE
from query_ast import ELLIPSE, PLUS
from query_ast import Color, Shape, Property
from query_ast import Is, Count
from query_ast import And, Or
from query_ast import Above, Below, Left, Right
from query_ast import Near

class TestQueryAST(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(TestQueryAST, self).__init__(*args, **kwargs)

    # Default properties
    self.red_ellipse = Property([Color(RED), Shape(ELLIPSE)])
    self.green_ellipse = Property([Color(GREEN), Shape(ELLIPSE)])
    self.blue_ellipse = Property([Color(BLUE), Shape(ELLIPSE)])

    self.red_plus = Property([Color(RED), Shape(PLUS)])
    self.green_plus = Property([Color(GREEN), Shape(PLUS)])
    self.blue_plus = Property([Color(BLUE), Shape(PLUS)])

    # Above queries
    self.red_ellipse_above_blue_plus = Is(Above(self.red_ellipse, self.blue_plus))
    self.green_ellipse_above_red_ellipse = Is(Above(self.green_ellipse, self.red_ellipse))

    # Below queries
    self.blue_plus_below_red_ellipse = Is(Below(self.blue_plus, self.red_ellipse))
    self.green_ellipse_below_red_ellipse = Is(Below(self.green_ellipse, self.red_ellipse))

    # Left queries
    self.red_ellipse_left_green_ellipse = Is(Left(self.red_ellipse, self.green_ellipse))
    self.green_ellipse_left_red_ellipse = Is(Left(self.green_ellipse, self.red_ellipse))

    # Right queries
    self.green_ellipse_right_red_ellipse = Is(Right(self.green_ellipse, self.red_ellipse))
    self.green_ellipse_right_blue_ellipse = Is(Right(self.green_ellipse, self.blue_ellipse))

    # Define some sample samples
    self.sample_1 = np.stack((np.array([[RED, GREEN], [BLUE, NULL_PROP]]),
                              np.array([[ELLIPSE, ELLIPSE], [PLUS, NULL_PROP]])))

  def testIsAboveQuery(self):
    self.assertEqual(self.red_ellipse_above_blue_plus.query(),
                     'Is red ellipse above blue plus')
    self.assertEqual(self.green_ellipse_above_red_ellipse.query(),
                     'Is green ellipse above red ellipse')

  def testIsAboveEval(self):
    self.assertEqual(self.red_ellipse_above_blue_plus.eval(self.sample_1), True)
    self.assertEqual(self.green_ellipse_above_red_ellipse.eval(self.sample_1), False)

  def testIsBelowQuery(self):
    self.assertEqual(self.blue_plus_below_red_ellipse.query(),
                     'Is blue plus below red ellipse')
    self.assertEqual(self.green_ellipse_below_red_ellipse.query(),
                     'Is green ellipse below red ellipse')

    # Define some sample samples
    self.sample_1 = np.stack((np.array([[RED, GREEN], [BLUE, NULL_PROP]]),
                              np.array([[ELLIPSE, ELLIPSE], [PLUS, NULL_PROP]])))

  def testIsAboveQuery(self):
    self.assertEqual(self.red_ellipse_above_blue_plus.query(),
                     'Is red ellipse above blue plus')
    self.assertEqual(self.green_ellipse_above_red_ellipse.query(),
                     'Is green ellipse above red ellipse')

  def testIsAboveEval(self):
    self.assertEqual(self.red_ellipse_above_blue_plus.eval(self.sample_1), True)
    self.assertEqual(self.green_ellipse_above_red_ellipse.eval(self.sample_1), False)

  def testIsBelowQuery(self):
    self.assertEqual(self.blue_plus_below_red_ellipse.query(),
                     'Is blue plus below red ellipse')
    self.assertEqual(self.green_ellipse_below_red_ellipse.query(),
                     'Is green ellipse below red ellipse')

  def testIsBelowEval(self):
    self.assertEqual(self.blue_plus_below_red_ellipse.eval(self.sample_1), True)
    self.assertEqual(self.green_ellipse_below_red_ellipse.eval(self.sample_1), False)

  def testIsLeftQuery(self):
    self.assertEqual(self.red_ellipse_left_green_ellipse.query(),
                     'Is red ellipse left of green ellipse')
    self.assertEqual(self.green_ellipse_left_red_ellipse.query(),
                     'Is green ellipse left of red ellipse')

  def testIsLeftEval(self):
    self.assertEqual(self.red_ellipse_left_green_ellipse.eval(self.sample_1), True)
    self.assertEqual(self.green_ellipse_left_red_ellipse.eval(self.sample_1), False)

  def testIsRightQuery(self):
    self.assertEqual(self.green_ellipse_right_red_ellipse.query(),
                     'Is green ellipse right of red ellipse')
    self.assertEqual(self.green_ellipse_right_blue_ellipse.query(),
                     'Is green ellipse right of blue ellipse')

  def testIsRightEval(self):
    self.assertEqual(self.green_ellipse_right_red_ellipse.eval(self.sample_1), True)
    self.assertEqual(self.green_ellipse_right_blue_ellipse.eval(self.sample_1), False)


if __name__ == '__main__':
  unittest.main()
