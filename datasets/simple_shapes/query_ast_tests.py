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

    # Define some default properties
    self.red_ellipse = Property([Color(RED), Shape(ELLIPSE)])
    self.green_ellipse = Property([Color(GREEN), Shape(ELLIPSE)])
    self.blue_ellipse = Property([Color(BLUE), Shape(ELLIPSE)])

    self.red_plus = Property([Color(RED), Shape(PLUS)])
    self.green_plus = Property([Color(GREEN), Shape(PLUS)])
    self.blue_plus = Property([Color(BLUE), Shape(PLUS)])

    # Define some above queries
    self.red_ellipse_above_blue_plus = Is(Above(self.red_ellipse, self.blue_plus))
    self.green_ellipse_above_red_ellipse = Is(Above(self.green_ellipse, self.red_ellipse))

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

if __name__ == '__main__':
  unittest.main()
