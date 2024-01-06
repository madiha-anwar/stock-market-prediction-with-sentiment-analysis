# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:24:35 2019

@author: Madiha
"""

from __future__ import print_function, division

import numpy as np
import pandas

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = '12'

from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

print(__doc__)


class Select_Features:
  def __init__(self) -> None:
    pass

  def run(self, dataset_file_path: str):
    names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend', 'SocialSentiment', 'FutureTrend']
    dataset = pandas.read_csv(dataset_file_path, names=names, skiprows=[0])
    print("\nDataset: %s" % dataset_file_path)

    print("\n==================== number of NaN values in each column ====================")
    print(dataset.isnull().sum())
    # remove null values
    #dataset.dropna(inplace=True)
    dataset.fillna(0,inplace=True)
    # ========================= encode categorical attribute values ==============
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()

    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            #le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])

    array = dataset.values
    X = array[:,0:8]
    Y = array[:,8]

    validation_size = 0.30
    num_folds = 10
    seed = 7

    pipe = Pipeline([
        # the reduce_dim stage is populated by the param_grid
        ('reduce_dim', None),
        ('classify', SVC())
    ])

    N_FEATURES_OPTIONS = [4, 5, 6, 7, 8]
    #C_OPTIONS = [1, 10, 100, 1000]

    C_OPTIONS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
    param_grid = [
        {
            'reduce_dim': [PCA(iterated_power=7)],
            'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
        {
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
    ]
    reducer_labels = ['PCA', 'SelectKBest (Chi2)']

    grid = GridSearchCV(pipe, cv=10, n_jobs=1, param_grid=param_grid)



    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
    test_size=validation_size, random_state=seed)


    scaler = MinMaxScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)


    grid.fit(rescaledX, Y_train)

    mean_scores = np.array(grid.cv_results_['mean_test_score'])
    print("Mean score: %s", mean_scores)
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))


    # select score for best C
    mean_scores = mean_scores.max(axis=0)*100
    bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
                  (len(reducer_labels) + 1) + .5)

    plt.figure()
    COLORS = 'bgrcmyk'
    for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
        plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

    plt.title("Comparing Feature Selection/Reduction Techniques")
    plt.xlabel('Number of Features Selected')
    plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
    plt.ylabel('Mean Testing Accuracy (%)')
    plt.ylim((0, 100))
    plt.legend(loc='upper left')

    plt.show()