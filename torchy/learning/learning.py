"""This file contains all classifier training and prediction functions"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from torchy import utils
import vigra
import h5py
import time
import pickle

import string
import random

random.seed()


def get_classifier(classifier_type, filename=None):
    if classifier_type == 'RandomForest':
        if filename is not None:
            print("loading sklearn forest from %s"%filename)
            return pickle.load(open(filename,"rb"))
        rf = RandomForestClassifier(max_depth=30, n_jobs=-1)
        print("Warning: random forest is trainged with random data...only for debug")
        rf.fit(np.random.rand(20,17), np.random.choice([0,1,2,3],size=20))
        return rf
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


def count_labels(classifier):
    if isinstance(classifier, vigra.learning.RandomForest):
        return classifier.labelCount()
    elif isinstance(classifier, RandomForestClassifier):
        return classifier.n_classes_
    else:
        raise NotImplementedError()


def image_prediction(classifier_type, classifier_file_name, features, request):
    s_list, f_list = utils.get_feature_index(request)
    classifier = get_classifier(classifier_type, classifier_file_name)
    export_shape = list(features.shape[2:])
    export_shape.append(count_labels(classifier))
    predshape = [len(s_list), np.product(features.shape[2:])]
    f = features[s_list, f_list].reshape(predshape).astype(np.float32)
    # write to file
    write_output(predict(classifier, f.T).reshape(export_shape), request, id_generator())
    return "done"


def predict(classifier, X_test):
    if isinstance(classifier, vigra.learning.RandomForest):
        return classifier.predictProbabilities(X_test)
    elif isinstance(classifier, RandomForestClassifier):
        return classifier.predict_proba(X_test)
    else:
        raise NotImplementedError()


def toh5(data, path, datapath='data'):
    with h5py.File(path, 'w') as f:
        f.create_dataset(datapath, data=data)


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def write_output(data, request, randstr):
    print("Writing...")
    toh5(data, request["output_file_name"] + randstr + '.h5', 'data')
    return