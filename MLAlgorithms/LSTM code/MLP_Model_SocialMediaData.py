# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:56:08 2019

@author: Madiha
"""

# ======= Stock prediction using deep learning  ==========

# ======= load libraries =================================
import pandas

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore")

from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = '10'
#rcParams['font.sans-serif'] = ['Tahoma']

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier

# ========= load dataset ============
dataset_file= "../data/prediction/sentiment/IBM.csv"
names       = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend', 'Twitter Sentiments','FutureTrend']
dataset     = pandas.read_csv(dataset_file, names=names, skiprows=[0])
print("\nDataset: %s" % dataset_file)

# ========= count the number of NaN values in each column ====
print("\n====== Number of NaN values in each column ========")
print(dataset.isnull().sum())
# remove null values
#dataset.dropna(inplace=True)
dataset.fillna(0,inplace=True)
# ========= print the first 5 rows of dataset =========
#print(dataset.head(5))
#print(dataset.tail(5))
#print(dataset[-50:])
#sys.exit()


# ======= encode categorical attribute values ========
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for column in dataset.columns:
    if dataset[column].dtype == type(object):
        #le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column].astype(str))

#print("\n============= dimension of dataset ====================")
#print(dataset.shape)

# ======== data types =====================

#print("\n======== data types =========")
#set_option('display.max_rows', 500)
#print(dataset.dtypes)

#print("\n======== statistical summary of dataset ==========")
#set_option('display.width', 100)
#set_option('precision', 3)
#description = dataset.describe()
#print(description)

print("\n======== Class distribution ========")
class_counts = dataset.groupby('FutureTrend').size()
print(class_counts)

#print("\n============== pearson correlations ====================")
#correlations = dataset.corr(method='pearson')
#print(correlations)

#print("\n==================== correlation matrix ====================")
#fig = pyplot.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
#fig.colorbar(cax)
#ticks = numpy.arange(0,14,1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
#pyplot.show()

#print("\n==================== univariate plots ====================")
#dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
#pyplot.show()

#print("\n==================== skew ====================")
#skew = dataset.skew()
#print(skew)

# density
#dataset.plot(kind='density', subplots=True, layout=(3,3), sharex=False, legend=False,
#fontsize=1)
#pyplot.show()

#print("\n==================== histogram ====================")
#dataset.hist()
#pyplot.show()

#print("\n==================== multivariate plots ====================")
#scatter_matrix(dataset)
#pyplot.show()

# ============= Split-out validation dataset ================
array           = dataset.values
X               = array[:,0:8]
Y               = array[:,8]
validation_size = 0.30
num_folds       = 10
seed            = 7
scoring         = 'accuracy'
 

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
test_size  =  validation_size, random_state=seed)

# ========= Spot-Check Algorithms =============
models = []
models.append(('MLP', MLPClassifier()))

# ========== evaluate each model in turn ============
print("\n======== Model evaluation on normal dataset ========")
results = []
names   = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())
    print(msg)

# ========================= Make predictions on validation dataset ===========
prediction_accuracy = []

# ========================= Standardize the dataset ==========================
pipelines = []
    
pipelines.append(('MLP', Pipeline([('Scaler', StandardScaler()),('MLP',
MLPClassifier())])))

# ================== prepare the models =======================================

# ================== prepare MLP model =======================================
print("\n======= MLP Accuracy on transformed validation dataset ==========")
scaler      = StandardScaler().fit(X_train)
rescaledX   = scaler.transform(X_train)
#model = MLPClassifier(alpha=0.0001, activation="tanh", solver="sgd", learning_rate="adaptive", hidden_layer_sizes=(5,5,5))
'''
how manay layers and nodes?
#1. use a robust test harness
2. deep networks perform best, i.e use more layers
3. test differet configurations
'''
model       = MLPClassifier(alpha=0.0001, activation="tanh", solver="adam", learning_rate="constant", hidden_layer_sizes=(100,100,100), max_iter=100)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX =   scaler.transform(X_validation)
predictions         =   model.predict(rescaledValidationX)


mlp_accuracy         =   accuracy_score(Y_validation, predictions)*100

print("Accuracy: %s" % mlp_accuracy)
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))

print("MLP Validation Accuracy: %s" % round(mlp_accuracy,2))
