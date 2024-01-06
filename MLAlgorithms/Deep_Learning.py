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

from pandas import set_option
from matplotlib import pyplot

from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = '10'
#rcParams['font.sans-serif'] = ['Tahoma']

from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

 
def get_pandas_column_names(file_path):
  import pandas
  dataframe = pandas.read_csv(file_path)
  return dataframe.columns.values.tolist()


class Deep_learning:
  def __init__(self) -> None:
    pass

  @classmethod
  def run(self, dataset_file_path: str):
    # ========================= load dataset =====================================
    names = get_pandas_column_names(dataset_file_path)
    dataset = pandas.read_csv(dataset_file_path, names=names, skiprows=[0])
    print("\nDataset: %s" % dataset_file_path)

    # ========================= encode categorical attribute values ==============
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()

    for column in dataset.columns:
      if dataset[column].dtype == type(object):
        #le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])

    # ========================= Split-out validation dataset =====================
    array = dataset.values
    X = array[:,0:len(names) - 1]
    Y = array[:,len(names) - 1]
    validation_size = 0.30
    num_folds = 10
    seed = 7
    scoring = 'accuracy'


    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
    test_size=validation_size, random_state=seed, shuffle=True)

    # ========================= Spot-Check Algorithms ============================
    models = []
    models.append(('MLP', MLPClassifier()))


    # ========================= evaluate each model in turn ======================
    print("\n==================== models evaluation ====================")
    results = []
    names = []
    for name, model in models:
      kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
      cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
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
        
    pipelines.append(('ScaledMLP', Pipeline([('Scaler', StandardScaler()),('MLP',
    MLPClassifier())])))

    results = []
    names = []
    for name, model in pipelines:
      kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
      cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
      results.append(cv_results)
      names.append(name)
      msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
      print(msg)


    print("\n==================== MLP Algorithm tuning ====================")

    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    parameter_space = {
      'hidden_layer_sizes': [(5,), (2,), (1,)],
      'activation': ['tanh', 'relu'],
      'solver': ['sgd', 'adam'],
      'alpha': [0.0001, 0.05],
      'learning_rate': ['constant','adaptive'],
    }

    model = MLPClassifier()
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    grid = GridSearchCV(estimator=model, param_grid=parameter_space, scoring=scoring, cv=kfold)
    grid_result = grid.fit(rescaledX, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))

