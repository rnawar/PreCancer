import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
from collections import Counter
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import math
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt


class Bagging:

    def __init__(self):
        self.data = None
        self.target_col = None
        self.k = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.base_models = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.n_models = None
        self.bagging_models = None
        self.data_id = None
        self.voting_classifier = None
        self.new_data = None
        self.new_data_id = None

    def replace_missing_values(self):
        has_nan = self.data.isna().sum().sum()
        if has_nan > 0:
            # replace missing values with mean
            self.data.fillna(self.data.mean(numeric_only=True).round(1), inplace=True)

    def preprocess_data(self):
        print("Now running preprocess_data function")
        self.data_id = self.data.iloc[:, [0]]
        X = self.data.drop(columns=[self.target_col, self.data.columns[0]])  # exclude target and first column
        y = self.data[self.target_col]

        # Encode target variable
        le = LabelEncoder()
        self.y = le.fit_transform(y)

        selector = SelectKBest(mutual_info_classif, k=self.k)  # use mutual information for feature selection
        X = selector.fit_transform(X, y)
        self.feature_names = list(
            self.data.drop(columns=[self.target_col, self.data.columns[0]]).columns[selector.get_support()])
        scaler = preprocessing.StandardScaler()
        self.X = scaler.fit_transform(X)

    def bagging(self):
        # print("Now running bagging function")
        models = []
        X_train, y_train = np.asarray(self.X_train), np.asarray(self.y_train)
        for model in self.base_models:
            for i in range(self.n_models):
                # create a bootstrap sample of the training data
                X_train_bag, y_train_bag = resample(X_train, y_train, replace=True, random_state=i)
                # train a base model on the bootstrap sample of the training data using LOOCV
                print("Running model {} ({})".format(i + 1, model))
                if model == 'SVC':
                    base_model = SVC(kernel='linear', random_state=i)
                elif model == 'NaiveBayes':
                    base_model = GaussianNB()
                elif model == 'RandomForest':
                    base_model = RandomForestClassifier(n_estimators=10, random_state=i)
                elif model == 'XGBoost':
                    base_model = xgb.XGBClassifier(random_state=i)
                else:
                    raise ValueError('Invalid base model')
                # perform LOOCV to train the model
                loocv = LeaveOneOut()
                for train_index, test_index in loocv.split(X_train_bag):
                    X_train_loocv, y_train_loocv = X_train_bag[train_index], y_train_bag[train_index]
                    base_model.fit(X_train_loocv, y_train_loocv)
                # plot feature importance for the base model
                #             plotBaggingFeatureImportance([base_model], X_train, y_train, feature_names)
                models.append(base_model)
        self.bagging_models = models

    def evaluate_bagging(self):
        self.voting_classifier = VotingClassifier(
            estimators=[(f"model_{i}", model) for i, model in enumerate(self.bagging_models)])
        self.voting_classifier.fit(self.X_train, self.y_train)
        y_pred = self.voting_classifier.predict(self.X_test)
        accuracy = round(accuracy_score(self.y_test, y_pred) * 100, 2)
        confusion = confusion_matrix(self.y_test, y_pred)
        TP = confusion[1, 1]  # true positive
        TN = confusion[0, 0]  # true negatives
        FP = confusion[0, 1]  # false positives
        FN = confusion[1, 0]  # false negatives
        sensitivity = round(TP / float(TP + FN), 2)
        specificity = round(TN / float(TN + FP), 2)
        auc = round((sensitivity + specificity) / 2, 2)

        # Create a dictionary with the performance metrics and confusion matrix
        results = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'confusion_matrix': confusion
        }

        return results

    def save_preprocessed_dataset(self):
        preprocessed_dataset = pd.DataFrame(self.X, columns=self.feature_names)
        preprocessed_dataset[self.target_col] = self.y
        preprocessed_dataset.insert(loc=0, column="ID_REF", value=self.data_id)
        # Return selected data DataFrame
        return preprocessed_dataset

    def set_new_data(self, new_data):
        self.new_data = new_data
        self.new_data_id = self.new_data.iloc[:, [0]]

    def predict_new_data(self):
        new_data = self.new_data.loc[:, self.feature_names]
        new_y_pred = self.voting_classifier.predict(new_data)
        output_file = pd.DataFrame(new_y_pred, columns=["Predicted Status"])
        output_file.insert(loc=0, column="ID_REF", value=self.new_data_id)
        return output_file

    def algorithm(self, data, target_col, k):
        self.data = data
        self.replace_missing_values()

        # Preprocess data
        self.target_col = target_col
        self.k = k
        self.preprocess_data()
        # self.save_preprocessed_dataset(X, y, feature_names, target_col)
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=42)

        # Train bagging model
        self.base_models = ['SVC', 'NaiveBayes', 'RandomForest', 'XGBoost']
        self.n_models = 3
        self.bagging()

        # Evaluate bagging model
        results = self.evaluate_bagging()
        return results
