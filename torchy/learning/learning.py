"""This file contains all classifier training and prediction functions"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

def getClassifier(classifier_type):
    if classifier_type == 'RandomForest':
        return RandomForestClassifier(max_depth=30)
    else:
        raise NotImplementedError()


def train(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)
    return classifier

def predict(classifier, X_test):
    return classifier.predict_proba(X_test)