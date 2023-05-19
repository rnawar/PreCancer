from unittest.mock import MagicMock

import pandas as pd
import unittest
from model import Bagging
import numpy as np


class BaggingIntegrationTestCase(unittest.TestCase):

    def setUp(self):
        # self.data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8], 'target': [0, 1, 0, 1]})
        self.data = pd.DataFrame({'col1': np.arange(1, 101), 'col2': np.arange(101, 201),
                                  'target': np.random.randint(0, 2, size=100)})
        self.target_col = 'target'
        self.k = 1
        self.bagging = Bagging()

    def test_algorithm(self):
        results = self.bagging.algorithm(self.data, self.target_col, self.k)
        self.assertIsInstance(results, dict)
        self.assertIn('accuracy', results)
        self.assertIn('sensitivity', results)
        self.assertIn('specificity', results)
        self.assertIn('auc', results)
        self.assertIn('confusion_matrix', results)

    def test_save_preprocessed_dataset(self):
        self.bagging.data = self.data
        self.bagging.X = [[-1.34164079, -1.34164079], [-0.4472136, -0.4472136], [0.4472136, 0.4472136],
                          [1.34164079, 1.34164079]]
        self.bagging.feature_names = ['col1', 'col2']
        self.bagging.y = [0, 1, 0, 1]
        self.bagging.data_id = pd.DataFrame({'ID_REF': ['id1', 'id2', 'id3', 'id4']})
        preprocessed_dataset = self.bagging.save_preprocessed_dataset()
        expected_columns = ['ID_REF', 'col1', 'col2', None]
        self.assertEqual(list(preprocessed_dataset.columns), expected_columns)
        self.assertEqual(len(preprocessed_dataset), 4)

    def test_set_new_data(self):
        new_data = pd.DataFrame({'col1': [2, 3, 4], 'col2': [6, 7, 8], 'target': [1, 0, 1]})
        self.bagging.set_new_data(new_data)
        self.assertEqual(self.bagging.new_data.shape, new_data.shape)
        self.assertTrue(self.bagging.new_data.equals(new_data))

    def test_predict_new_data(self):
        results = self.bagging.algorithm(self.data, self.target_col, self.k)
        self.bagging.data = self.data
        self.bagging.feature_names = ['col2']
        self.bagging.new_data = pd.DataFrame({'col2': [6, 7]})
        self.bagging.new_data_id = pd.DataFrame({'ID_REF': [101, 102]})
        output_file = self.bagging.predict_new_data()
        expected_columns = ['ID_REF', 'Predicted Status']
        self.assertEqual(list(output_file.columns), expected_columns)
        self.assertEqual(len(output_file), len(self.bagging.new_data))


if __name__ == '__main__':
    unittest.main()
