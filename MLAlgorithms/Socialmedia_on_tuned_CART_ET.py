# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:56:08 2019

@author: Madiha
"""

# ======================= Stock prediction using social media  ======================

# ========================= load libraries ===================================
import pandas

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore")


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

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

# ========================= load dataset =====================================
dataset_file = "datasets/Social media/HPQ/9-micro-2.0y.csv"
#dataset_file = "datasets/News and Social media/LSE/10-micro-ns-2.0y.csv"
names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend', 'SocialSentiment', 'FutureTrend']
#names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend','NewsSentiment', 'SocialSentiment', 'FutureTrend']
dataset = pandas.read_csv(dataset_file, names=names, skiprows=[0])
print("\nDataset: %s" % dataset_file)


# ========================= count the number of NaN values in each column ====
print("\n==================== number of NaN values in each column ====================")
print(dataset.isnull().sum())
# remove null values
#dataset.dropna(inplace=True)
dataset.fillna(0,inplace=True)
# ========================= print the first 5 rows of dataset ================

# ========================= encode categorical attribute values ==============
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for column in dataset.columns:
    if dataset[column].dtype == type(object):
        #le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column].astype(str))


print("\n==================== class distribution ====================")
class_counts = dataset.groupby('FutureTrend').size()
print(class_counts)


# ========================= Split-out validation dataset =====================
array = dataset.values
X = array[:,0:8]
Y = array[:,8]
validation_size = 0.30
num_folds = 10
seed = 7
scoring = 'accuracy'


X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
test_size=validation_size, random_state=seed)

# ========================= Spot-Check Algorithms ============================
models = []
models.append(('CART', DecisionTreeClassifier()))
models.append(('ET', ExtraTreesClassifier()))

# ========================= Make predictions on validation dataset ===========
prediction_accuracy = []


# ========================= Standardize the dataset ==========================
pipelines = []
    
pipelines.append(('CART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeClassifier())])))

pipelines.append(('ET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesClassifier())])))

# ================== Models accuracy before params tuning =======================================
print("\n==================== Performance before params tuning====================")

#------------------ CART Model ------------------
cart_accuracy=[]
cart         = DecisionTreeClassifier()

scaler      = StandardScaler().fit(X_train)
rescaledX   = scaler.transform(X_train)

cart.fit(rescaledX, Y_train)
rescaledValidationX = scaler.transform(X_validation)

cart_predictions = cart.predict(rescaledValidationX)
cart_accuracy    = accuracy_score(Y_validation, cart_predictions)*100

print("\nCART Accuracy: %s" % round(cart_accuracy,2))
print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, cart_predictions))
print("\nClassification report:")
print(classification_report(Y_validation, cart_predictions))
#prediction_accuracy.append(cart_accuracy)


#------------------ ET Model ------------------
et_accuracy=[]
et         = ExtraTreesClassifier()

scaler      = StandardScaler().fit(X_train)
rescaledX   = scaler.transform(X_train)

et.fit(rescaledX, Y_train)
rescaledValidationX = scaler.transform(X_validation)

et_predictions= et.predict(X_validation)
et_accuracy   = accuracy_score(Y_validation, et_predictions)*100

print("\nET Accuracy: %s" % round(et_accuracy,2))
print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, et_predictions))
print("\nClassification report:")
print(classification_report(Y_validation, et_predictions))


print("\n==================== Performance after params tuning====================")
# ================== prepare CART model =======================================
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = DecisionTreeClassifier(criterion= 'entropy', max_features='log2', min_samples_split=2, random_state=123, min_samples_leaf=1)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
cart_accuracy=accuracy_score(Y_validation, predictions)*100

print("\nCART Accuracy: %s" % round(cart_accuracy,2))
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))

# ================== prepare ET model =======================================
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = ExtraTreesClassifier(n_jobs=-1, min_samples_leaf=1, n_estimators=20, random_state=123, criterion='gini', min_samples_split=2)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
et_accuracy=accuracy_score(Y_validation, predictions)*100

print("\nET Accuracy: %s" % round(et_accuracy,2))
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))