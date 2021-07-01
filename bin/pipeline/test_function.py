import unittest
import pandas as pd
import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))

from ml_pipeline import XGBoost_Model_Predict

class TestDataFunction(unittest.TestCase):
    
    def Test_ML_Pipeline(self):
        
        r1 = XGBoost_Model_Predict()
        r2 = 0.8006
        self.assertEqual(r1,r2)
        
if __name__ == '__main__':
    unittest.main()
        
        
