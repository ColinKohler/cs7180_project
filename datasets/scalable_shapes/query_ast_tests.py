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
    
    self.red = Color(RED)
    self.blue = Color(BLUE)
    self.green = Color(GREEN)
    
    self.ellipse = Shape(ELLIPSE)
    self.plus = Shape(PLUS)

    # Is queries
    self.is_red_ellipse = Is(self.red_ellipse)
    self.is_green_plus = Is(self.green_plus)

    # Above queries
    self.red_ellipse_above_blue_plus = Is(Above(self.red_ellipse, self.blue_plus))
    self.green_ellipse_above_red_ellipse = Is(Above(self.green_ellipse, self.red_ellipse))
    self.ellipse_above_plus = Above(self.ellipse, self.plus)
    self.green_above_blue = Above(self.green, self.blue)
    
    # Below queries
    self.blue_plus_below_red_ellipse = Is(Below(self.blue_plus, self.red_ellipse))
    self.green_ellipse_below_red_ellipse = Is(Below(self.green_ellipse, self.red_ellipse))

    # Left queries
    self.red_ellipse_left_green_ellipse = Is(Left(self.red_ellipse, self.green_ellipse))
    self.green_ellipse_left_red_ellipse = Is(Left(self.green_ellipse, self.red_ellipse))

    # Right queries
    self.green_ellipse_right_red_ellipse = Is(Right(self.green_ellipse, self.red_ellipse))
    self.green_ellipse_right_blue_ellipse = Is(Right(self.green_ellipse, self.blue_ellipse))

    # Count queries
    self.count_blue = Count(self.blue)
    self.count_ellipse = Count(self.ellipse)
    self.count_red_ellipse_above_blue_plus = Count(self.red_ellipse_above_blue_plus)
    self.count_ellipse_above_plus = Count(self.ellipse_above_plus)
    self.count_green_above_blue = Count(self.green_above_blue)

    # Near queries
    self.is_red_near_blue = Is(Near(Color(RED),Color(BLUE)))

    # Define some sample samples
    self.sample_1 = np.stack((np.array([[RED, GREEN], [BLUE, NULL_PROP]]),
                              np.array([[ELLIPSE, ELLIPSE], [PLUS, NULL_PROP]])))
    self.sample_2 = np.stack((np.array([[RED, GREEN], [BLUE, BLUE]]),
                              np.array([[ELLIPSE, ELLIPSE], [PLUS, PLUS]])))
    self.sample_3 = np.stack((np.array([[RED, NULL_PROP, NULL_PROP], [NULL_PROP, NULL_PROP, NULL_PROP], [NULL_PROP, NULL_PROP, BLUE]]),
                              np.array([[ELLIPSE, NULL_PROP, NULL_PROP], [NULL_PROP, NULL_PROP, NULL_PROP], [NULL_PROP, NULL_PROP, ELLIPSE]])))
    self.sample_4 = np.stack((np.array([[RED, NULL_PROP, NULL_PROP], [NULL_PROP, BLUE, NULL_PROP], [NULL_PROP, NULL_PROP, NULL_PROP ]]),
                              np.array([[ELLIPSE, NULL_PROP, NULL_PROP], [NULL_PROP, ELLIPSE, NULL_PROP], [NULL_PROP, NULL_PROP, NULL_PROP]])))                              

  def testIsPropertyQuery(self):
    self.assertEqual(self.is_red_ellipse.query(),
                     'Is red ellipse')
    self.assertEqual(self.is_green_plus.query(),
                     'Is green plus')

  def testIsPropertyEval(self):
    self.assertEqual(self.is_red_ellipse.eval(self.sample_1), True)
    self.assertEqual(self.is_green_plus.eval(self.sample_1), False)

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

  def testCountEval(self):
    self.assertEqual(self.count_blue.eval(self.sample_2), 2)
    self.assertEqual(self.count_ellipse.eval(self.sample_2), 2)
  
  def testCountAboveEval(self):
    self.assertEqual(self.count_red_ellipse_above_blue_plus.eval(self.sample_2),1)
    self.assertEqual(self.count_ellipse_above_plus.eval(self.sample_2),2)
    self.assertEqual(self.count_green_above_blue.eval(self.sample_2),1)
    
  def testNearEval(self):
    self.assertEqual(self.is_red_near_blue.eval(self.sample_3),False)
    self.assertEqual(self.is_red_near_blue.eval(self.sample_4),True)

if __name__ == '__main__':
  unittest.main()
