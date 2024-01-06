# -*- coding: utf-8 -*-
"""

@author: Madiha
"""
# ======================= Combining best classifiers  ======================

# ========================= load libraries ===================================
import numpy as np
import pandas
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler

def get_pandas_column_names(file_path):
  import pandas
  dataframe = pandas.read_csv(file_path)
  return dataframe.columns.values.tolist()


class Voting_Ensemble:
  def __init__(self) -> None:
    pass

  @classmethod
  def run(self,  dataset_file_path: str):
    # ========================= load dataset =====================================
    names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend', 'SocialSentiment', 'FutureTrend']
    names = get_pandas_column_names(dataset_file_path)
    dataset = pandas.read_csv(dataset_file_path, names=names, skiprows=[0])
    print("\nDataset: %s" % dataset_file_path)

    # ========================= count the number of NaN values in each column ====
    #print("\n==================== number of NaN values in each column ====================")
    #print(dataset.isnull().sum())
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

    print("\n==================== dimension of dataset ====================")
    print(dataset.shape)

    array = dataset.values
    X = array[:,0:len(names) - 1]
    Y = array[:,len(names) - 1]
    validation_size = 0.30
    num_folds = 10
    seed = 7
    scoring = 'accuracy'

    # ========================= splitting dataset into training and testing datasets ==============
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
    test_size=validation_size, random_state=seed)

    # ========================= classifiers to be combined ==============
    RF_classifier=RandomForestClassifier(n_jobs=-1, min_samples_leaf=1, n_estimators=20, random_state=123, criterion='entropy', min_samples_split=6)
    ET_classifier=ExtraTreesClassifier(n_jobs=-1, min_samples_leaf=1, n_estimators=15, random_state=123, criterion='gini', min_samples_split=3)
    GBM_classifier=GradientBoostingClassifier(n_estimators=400)


    # ========================= RF classifiers ==============
    eRF_classifier = VotingClassifier(estimators=[
            ('RF', RF_classifier), ('ET', ET_classifier), ('GBM', GBM_classifier)], voting='hard')

        
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)

    eRF_classifier = RF_classifier.fit(rescaledX, Y_train)

    rescaledValidationX = scaler.transform(X_validation)
    RF_predictions=eRF_classifier.predict(rescaledValidationX)
    #print(eclf1.predict(X_validation))
    RF_accuracy=accuracy_score(Y_validation, RF_predictions)*100
    print("RF Accuracy: %s" % RF_accuracy)
    print("Confusion matrix: %s" % confusion_matrix(Y_validation, RF_predictions))
    print("Classification report: %s" % classification_report(Y_validation, RF_predictions))
    #np.array_equal(eRF_classifier.named_estimators_.RF.predict(X_train),
                  #eRF_classifier.named_estimators_['RF'].predict(X_train))
                  
                  
    # ========================= ET classifiers ==============
    eET_classifier = VotingClassifier(estimators=[
            ('RF', RF_classifier), ('ET', ET_classifier), ('GBM', GBM_classifier)],
            voting='hard')

    eET_classifier = eET_classifier.fit(rescaledX, Y_train)

    ET_predictions=eET_classifier.predict(rescaledValidationX)
    #print(eclf2.predict(X_validation))
    ET_accuracy=accuracy_score(Y_validation, ET_predictions)*100
    print("ET Accuracy: %s" % ET_accuracy)
    print("Confusion matrix: %s" % confusion_matrix(Y_validation, ET_predictions))
    print("Classification report: %s" % classification_report(Y_validation, ET_predictions))


    # ========================= GBM classifiers ==============
    eGBM_classifier = VotingClassifier(estimators=[
          ('RF', RF_classifier), ('ET', ET_classifier), ('GBM', GBM_classifier)],
          voting='hard')
    eGBM_classifier = eGBM_classifier.fit(rescaledX, Y_train)
    GBM_predictions=eGBM_classifier.predict(rescaledValidationX)
    #print(eclf3.predict(X_validation))
    GBM_accuracy=accuracy_score(Y_validation, GBM_predictions)*100
    print("GBM Accuracy: %s" % GBM_accuracy)
    print("Confusion matrix: %s" % confusion_matrix(Y_validation, GBM_predictions))
    print("Classification report: %s" % classification_report(Y_validation, GBM_predictions))
    #print(eclf3.transform(X).shape)

    print("\n==================== Accuracies ====================")
    print("RF: %s" % round(RF_accuracy,2))
    print("ET: %s" % round(ET_accuracy,2))
    print("GBM: %s" % round(GBM_accuracy,2))