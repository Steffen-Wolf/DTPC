"""This file contains all classifier training and prediction functions"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import vigra

def get_classifier(classifier_type, filename=None):
    if classifier_type == 'RandomForest':
        return RandomForestClassifier(max_depth=30)
    elif classifier_type == 'VigraRF':
        if filename is not None:
            return vigra.learning.RandomForest(filename, 'Forest0000')
        return vigra.learning.RandomForest()
    else:
        raise NotImplementedError()

def train(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)
    return classifier

def load_classifier(classifier):

    return classifier

def predict(classifier, X_test):
    if isinstance(classifier, vigra.learning.RandomForest):
        return classifier.predictProbabilities
    elif isinstance(classifier, sklearn.ensemble.RandomForestClassifier):
        return classifier.predict_proba(X_test)
    else:
        raise NotImplementedError()