import unittest

from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

from model import Bagging
import numpy as np
import pandas as pd


class BaggingStatementCoverageTest(unittest.TestCase):

    def test_replace_missing_values(self):
        # Create test data with missing values
        data = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]})
        target_col = 'Target'
        k = 3

        # Initialize Bagging instance
        bagging = Bagging()
        bagging.data = data
        bagging.target_col = target_col
        bagging.k = k

        # Call the replace_missing_values method
        bagging.replace_missing_values()

        # Assert that there are no missing values in the data
        self.assertEqual(bagging.data.isna().sum().sum(), 0)

    def test_preprocess_data_1(self):
        # Create test data
        data = pd.DataFrame({'ID': [1, 2, 3, 4, 5], 'B': [4, 5, 6, 7, 8], 'Target': [0, 1, 0, 1, 1]})

        # Repeat the existing rows to have at least 10 data points
        while len(data) < 100:
            data = pd.concat([data, data])
        target_col = 'Target'
        k = 1

        # Initialize Bagging instance
        bagging = Bagging()
        bagging.data = data
        bagging.target_col = target_col
        bagging.k = k

        # Call the preprocess_data method
        bagging.preprocess_data()

        # Assert that X, y, and feature_names are properly assigned
        self.assertIsNotNone(bagging.X)
        self.assertIsNotNone(bagging.y)
        self.assertIsNotNone(bagging.feature_names)

    def test_preprocess_data_2(self):
        # Create test data
        data = pd.DataFrame({'ID': [1, 2, 3], 'B': [4, 5, 6], 'Target': ['Yes', 'No', 'Yes']})

        # Initialize Bagging instance
        bagging = Bagging()
        bagging.data = data
        bagging.target_col = 'Target'
        bagging.k = 1

        # Call the preprocess_data method
        bagging.preprocess_data()

        # Assert that the feature_names attribute is set correctly
        expected_feature_names = ['B']
        self.assertListEqual(bagging.feature_names, expected_feature_names)

        # Assert that the X attribute is transformed correctly
        expected_X = np.array([[-1.22474487],
                               [0.],
                               [1.22474487]])
        np.testing.assert_array_almost_equal(bagging.X, expected_X)

    def test_bagging(self):
        # Create test data
        np.random.seed(123)

        # Create test data
        data = pd.DataFrame({'ID': np.arange(1, 101), 'B': np.arange(101, 201),
                             'Target': np.random.randint(0, 2, size=100)})

        target_col = 'Target'
        k = 1

        # Initialize Bagging instance
        bagging = Bagging()
        bagging.data = data
        bagging.target_col = target_col
        bagging.k = k

        # Call the preprocess_data method (assuming it has been tested separately)
        bagging.preprocess_data()

        # Set up necessary attributes for bagging
        bagging.base_models = ['SVC', 'NaiveBayes','AAA']
        bagging.n_models = 1
        bagging.X_train = bagging.X
        bagging.y_train = bagging.y
        # Call the bagging method
        bagging.bagging()

        # Assert that the bagging_models attribute is populated
        self.assertIsNotNone(bagging.bagging_models)

    def test_evaluate_bagging(self):
        # Create test data
        data = pd.DataFrame({'ID': np.arange(1, 101), 'B': np.arange(101, 201),
                             'Target': np.random.randint(0, 2, size=100)})
        target_col = 'Target'
        k = 1

        # Initialize Bagging instance
        bagging = Bagging()
        bagging.data = data
        bagging.target_col = target_col
        bagging.k = k

        # Call the preprocess_data method (assuming it has been tested separately)
        bagging.preprocess_data()

        # Set up necessary attributes for evaluation
        bagging.base_models = ['SVC', 'NaiveBayes','AAA']
        bagging.n_models = 1
        bagging.X_train = bagging.X
        bagging.y_train = bagging.y
        bagging.X_test = bagging.X
        bagging.y_test = bagging.y
        bagging.bagging()
        # Call the evaluate_bagging method
        results = bagging.evaluate_bagging()

        # Assert that the results dictionary contains the expected keys
        expected_keys = ['accuracy', 'sensitivity', 'specificity', 'auc', 'confusion_matrix']
        self.assertListEqual(list(results.keys()), expected_keys)

    def test_save_preprocessed_dataset(self):
        # Create test data
        data = pd.DataFrame({'ID': [1, 2, 3], 'B': [4, 5, 6], 'Target': [0, 1, 0]})
        target_col = 'Target'
        k = 1

        # Initialize Bagging instance
        bagging = Bagging()
        bagging.data = data
        bagging.target_col = target_col
        bagging.k = k

        # Call the preprocess_data method (assuming it has been tested separately)
        bagging.preprocess_data()

        # Call the save_preprocessed_dataset method
        preprocessed_dataset = bagging.save_preprocessed_dataset()

        # Assert that the preprocessed_dataset DataFrame is not empty
        self.assertFalse(preprocessed_dataset.empty)


    def test_set_new_data(self):
        # Create test data
        new_data = pd.DataFrame({'A': [4, 5, 6], 'B': [7, 8, 9]})

        # Initialize Bagging instance
        bagging = Bagging()

        # Call the set_new_data method
        bagging.set_new_data(new_data)

        # Assert that the new_data attribute is set correctly
        self.assertEqual(bagging.new_data.shape, new_data.shape)

    def test_predict_new_data(self):

        # Create test data
        data = pd.DataFrame({'ID': np.arange(1, 101), 'B': np.arange(101, 201),
                             'Target': np.random.randint(0, 2, size=100)})
        target_col = 'Target'
        k = 1

        # Initialize Bagging instance
        bagging = Bagging()
        bagging.data = data
        bagging.target_col = target_col
        bagging.k = k

        # Call the preprocess_data method (assuming it has been tested separately)
        bagging.preprocess_data()

        # Set up necessary attributes for evaluation
        bagging.base_models = ['SVC', 'NaiveBayes','AAA']
        bagging.n_models = 1
        bagging.X_train = bagging.X
        bagging.y_train = bagging.y
        bagging.X_test = bagging.X
        bagging.y_test = bagging.y
        bagging.bagging()
        bagging.evaluate_bagging()

        # Create test data
        new_data = pd.DataFrame({'ID': [4, 5, 6], 'B': [7, 8, 9]})

        # Initialize Bagging instance
        bagging.new_data = new_data

        # Set up necessary attributes for prediction
        bagging.feature_names = ['B']

        # Call the predict_new_data method
        output_file = bagging.predict_new_data()

        # Assert that the output_file DataFrame is not empty
        self.assertFalse(output_file.empty)

    def test_algorithm(self):
        # Create test data
        data = pd.DataFrame({'ID': [1, 2, 3], 'B': [4, 5, 6], 'Target': [0, 1, 0]})

        # Repeat the existing rows to have at least 10 data points
        while len(data) < 100:
            data = pd.concat([data, data])

        target_col = 'Target'
        k = 1

        # Initialize Bagging instance
        bagging = Bagging()

        # Call the algorithm method with the test data
        results = bagging.algorithm(data, target_col, k)

        # Assert that the results dictionary contains the expected keys
        expected_keys = ['accuracy', 'sensitivity', 'specificity', 'auc', 'confusion_matrix']
        self.assertListEqual(list(results.keys()), expected_keys)



if __name__ == '__main__':
    unittest.main()
