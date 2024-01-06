# -*- coding: utf-8 -*-
"""

@author: Madiha
"""
# ========================= Algorithms Tuning ================================

# ========================= load libraries ===================================
import pandas
import numpy

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from matplotlib import pyplot

from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = '10'
#rcParams['font.sans-serif'] = ['Tahoma']


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


from sklearn.tree import DecisionTreeClassifier


class CART_Tuning:
  def __init__(self) -> None:
    pass

  def run(self, dataset_file_path: str):
    # ========================= load dataset =====================================
    names       = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend', 'SocialSentiment', 'FutureTrend']
    dataset     = pandas.read_csv(dataset_file_path, names=names, skiprows=[0])
    print("\nDataset: %s" % dataset_file_path)

    # ========================= encode categorical attribute values ==============
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()

    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            #le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])

    # ========================= Split-out validation dataset =====================
    array          = dataset.values
    X              = array[:,0:8]
    Y              = array[:,8]
    validation_size= 0.30
    num_folds      = 10
    seed           = 7
    scoring        = 'accuracy'


    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
    test_size=validation_size, random_state=seed)

    # ========================= Spot-Check Algorithms ============================
    models = []

    models.append(('CART', DecisionTreeClassifier()))

    # ========================= evaluate each model in turn ======================
    print("\n==================== models evaluation ====================")
    results= []
    names  = []
    for name, model in models:
        kfold     = KFold(n_splits=num_folds, random_state=seed)
        cv_results= cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    # ========================= Compare Algorithms graphically ===================
        
    print("\n==================== compare models graphically ====================")
    fig = pyplot.figure()
    fig.suptitle('Accuracy comparison of classifiers')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    pyplot.show()

    # ========================= Standardize the dataset ==========================
    print("\n==================== Evaluation results on standardized dataset ====================")
    pipelines = []
        
        
    pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
    DecisionTreeClassifier())])))

    results = []
    names = []
    for name, model in pipelines:
        kfold     = KFold(n_splits=num_folds, random_state=seed)
        cv_results= cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


    print("\n==================== Tune scaled CART ====================")
    scaler   = StandardScaler().fit(X_train)
    rescaledX= scaler.transform(X_train)
    #criteria = ['gini', 'entropy']

    params = {'criterion':['gini','entropy'],
              'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_split': [1.0,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
              'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],
              'random_state':[123],
              'splitter':['best', 'random']}

    #param_grid = dict(criterian=criteria)
    model = DecisionTreeClassifier()
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid  = GridSearchCV(estimator=model, param_grid=params, scoring=scoring, cv=kfold)
    grid_result = grid.fit(rescaledX, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds   = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #   print("%f (%f) with: %r" % (mean, stdev, param))
