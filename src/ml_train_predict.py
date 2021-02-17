"""
functions:
training(x_train, y_train, feature_type, time_df, index)
prediction(x_test, feature_type, df, time_df, index)
"""

from constants import *
import os
import joblib
import time
from data_saver import save_ml_model
from data_loader import load_ml_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,\
    AdaBoostClassifier, GradientBoostingClassifier


def training(x_train, y_train, feature_type, time_df, index):
    """
    function training(x_train, y_train, feature_type, time_df, index)
    sets classificators params, trains the model and saves it
    """
    C = 1.0

    from sklearn import svm

    classifiers = [LogisticRegression(random_state=RANDOM_SEED),
                   DecisionTreeClassifier(random_state=RANDOM_SEED),
                   KNeighborsClassifier(n_neighbors=300),
                   svm.LinearSVC(C=C, max_iter=10000),
                   RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=200,
                                          criterion='gini'),
                   AdaBoostClassifier(n_estimators=100, random_state=RANDOM_SEED),
                   GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)]

    # train classifiers and save trained models
    for i, clf in enumerate(classifiers):
        t0 = time.time()
        clf.fit(x_train, y_train)
        t = time.time()
        time_df.loc[CATEGORIES[index]][COL_LIST[i]] += (t - t0)
        save_ml_model(clf, COL_LIST[i] + feature_type)


def prediction(x_test, feature_type, df, time_df, index):
    """
    function prediction(x_test, feature_type, df, time_df, index)
    loads trained model and predicts the result
    """
    # load  trained models
    classifiers = []
    for i, clf_names in enumerate(COL_LIST):
        classifiers.append(load_ml_model(COL_LIST[i] + feature_type))

    # predict results and save time
    for i, clf in enumerate(classifiers):
        t0 = time.time()
        df[COL_LIST[i]] = clf.predict(x_test)
        t = time.time()
        time_df.loc[CATEGORIES[index]][COL_LIST[i]] += (t - t0)/len(x_test)
