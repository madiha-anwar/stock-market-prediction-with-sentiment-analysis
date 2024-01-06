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


# ========================= load dataset =====================================
dataset_file_path = "datasets/News and Social media/LSE/10-micro-ns-2.0y.csv"
names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend', 'SocialSentiment', 'FutureTrend']
#names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend','NewsSentiment', 'SocialSentiment', 'FutureTrend']
dataset = pandas.read_csv(dataset_file_path, names=names, skiprows=[0])
print("\nDataset: %s" % dataset_file_path)


# ========================= count the number of NaN values in each column ====
print("\n==================== number of NaN values in each column ====================")
print(dataset.isnull().sum())
# remove null values
#dataset.dropna(inplace=True)
dataset.fillna(0,inplace=True)
# ========================= print the first 5 rows of dataset ================

print(dataset.tail(5))



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
models.append(('SVM', SVC()))
models.append(('MLP', MLPClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('AB', AdaBoostClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('ET', ExtraTreesClassifier()))

# ========================= evaluate each model in turn ======================
print("\n==================== models evaluation on normal dataset =================")
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())
    print(msg)
    
# ========================= Compare Algorithms graphically ===================
    
print("\n==================== compare models graphically ====================")
fig = pyplot.figure()
fig.suptitle('Algorithms Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# ========================= Make predictions on validation dataset ===========
print("\n==================== Make predictions on validation dataset ====================")
prediction_accuracy = []


# ========================= Standardize the dataset ==========================
print("\n==================== Evaluation results on standardized dataset ====================")
pipelines = []
    
pipelines.append(('SVM', Pipeline([('Scaler', StandardScaler()),('SVM',
SVC())])))
    
pipelines.append(('MLP', Pipeline([('Scaler', StandardScaler()),('MLP',
MLPClassifier())])))
    
pipelines.append(('KNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsClassifier())])))
    
pipelines.append(('CART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeClassifier())])))
    
pipelines.append(('LDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))

pipelines.append(('AB', Pipeline([('Scaler', StandardScaler()),('AB',
AdaBoostClassifier())])))
    
pipelines.append(('GBM', Pipeline([('Scaler', StandardScaler()),('GBM',
GradientBoostingClassifier())])))
    
pipelines.append(('RF', Pipeline([('Scaler', StandardScaler()),('RF',
RandomForestClassifier())])))

pipelines.append(('ET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesClassifier())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())
    print(msg)

fig = pyplot.figure()
fig.suptitle('Algorithms Comparison on Transformed and Standardized Dataset')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
# ================== Models accuracy before params tuning =======================================
print("\n==================== Accuracies before params tuning====================")
#------------------ SVM Model------------------
svm_accuracy=[]
svc         = SVC()

scaler      = StandardScaler().fit(X_train)
rescaledX   = scaler.transform(X_train)

svc.fit(rescaledX, Y_train)

rescaledValidationX = scaler.transform(X_validation)
svc_predictions     = svc.predict(rescaledValidationX)
svm_accuracy        = accuracy_score(Y_validation, svc_predictions)*100


model    = SVC(kernel="rbf", C=0.5)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
predictions         = model.predict(rescaledValidationX)
svm_accuracy_apt=accuracy_score(Y_validation, predictions)*100

#------------------ MLP Model ------------------
mlp_accuracy=[]
mlp         = MLPClassifier()

scaler      = StandardScaler().fit(X_train)
rescaledX   = scaler.transform(X_train)

mlp.fit(rescaledX, Y_train)
rescaledValidationX = scaler.transform(X_validation)

mlp_predictions = mlp.predict(rescaledValidationX)
mlp_accuracy    = accuracy_score(Y_validation, mlp_predictions)*100
prediction_accuracy.append(mlp_accuracy)

#------------------ KNN Model ------------------
knn_accuracy=[]
knn         = KNeighborsClassifier()

scaler      = StandardScaler().fit(X_train)
rescaledX   = scaler.transform(X_train)


knn.fit(rescaledX, Y_train)
rescaledValidationX = scaler.transform(X_validation)

knn_predictions = knn.predict(rescaledValidationX)
knn_accuracy    = accuracy_score(Y_validation, knn_predictions)*100
prediction_accuracy.append(knn_accuracy)


#------------------ CART Model ------------------
cart_accuracy=[]
cart         = DecisionTreeClassifier()

scaler      = StandardScaler().fit(X_train)
rescaledX   = scaler.transform(X_train)

cart.fit(rescaledX, Y_train)
rescaledValidationX = scaler.transform(X_validation)

cart_predictions = cart.predict(rescaledValidationX)
cart_accuracy    = accuracy_score(Y_validation, cart_predictions)*100
prediction_accuracy.append(cart_accuracy)


#------------------ LDA Accuracy on validation dataset ------------------
lda_accuracy=[]
lda         = LinearDiscriminantAnalysis()

scaler      = StandardScaler().fit(X_train)
rescaledX   = scaler.transform(X_train)

lda.fit(rescaledX, Y_train)
rescaledValidationX = scaler.transform(X_validation)

lda_predictions = lda.predict(rescaledValidationX)
lda_accuracy    = accuracy_score(Y_validation, lda_predictions)*100
prediction_accuracy.append(lda_accuracy)


#------------------ AB Model ------------------
ab_accuracy=[]
ab         = AdaBoostClassifier()

scaler      = StandardScaler().fit(X_train)
rescaledX   = scaler.transform(X_train)

ab.fit(rescaledX, Y_train)
rescaledValidationX = scaler.transform(X_validation)

ab_predictions= ab.predict(rescaledValidationX)
ab_accuracy   = accuracy_score(Y_validation, ab_predictions)*100


#------------------ GBM Model ------------------
gbm_accuracy=[]
gbm         = GradientBoostingClassifier()

scaler      = StandardScaler().fit(X_train)
rescaledX   = scaler.transform(X_train)

gbm.fit(rescaledX, Y_train)
rescaledValidationX = scaler.transform(X_validation)

gbm_predictions = gbm.predict(rescaledValidationX)
gbm_accuracy    = accuracy_score(Y_validation, gbm_predictions)*100


#------------------ RF Model ------------------
rf_accuracy=[]
rf         = RandomForestClassifier()

scaler      = StandardScaler().fit(X_train)
rescaledX   = scaler.transform(X_train)

rf.fit(rescaledX, Y_train)
rescaledValidationX = scaler.transform(X_validation)

rf_predictions      = rf.predict(rescaledValidationX)
rf_accuracy         = accuracy_score(Y_validation, rf_predictions)*100


#------------------ ET Model ------------------
et_accuracy=[]
et         = ExtraTreesClassifier()

scaler      = StandardScaler().fit(X_train)
rescaledX   = scaler.transform(X_train)

et.fit(rescaledX, Y_train)
rescaledValidationX = scaler.transform(X_validation)

et_predictions= et.predict(X_validation)
et_accuracy   = accuracy_score(Y_validation, et_predictions)*100



print("\n-------------------- Accuracies before parameters tuning --------------------") 
print("\nSVM Accuracy: %s" % round(svm_accuracy,2))
print("\nSVM Accuracy apt: %s" % round(svm_accuracy_apt,2))

print("\nMLP Accuracy: %s" % round(mlp_accuracy,2))
print("\nKNN Accuracy: %s" % round(knn_accuracy,2))
print("\nCART Accuracy: %s" % round(cart_accuracy,2))
print("\nLDA Accuracy: %s" % round(lda_accuracy,2))
print("\nAB Accuracy: %s" % round(ab_accuracy,2))
print("\nGBM Accuracy: %s" % round(gbm_accuracy,2))
print("\nRF Accuracy: %s" % round(rf_accuracy,2))
print("\nET Accuracy: %s" % round(et_accuracy,2))


print("\n==================== Accuracies after params tuning====================")

scaler   = StandardScaler().fit(X_train)
rescaledX= scaler.transform(X_train)
model    = SVC(kernel="rbf", C=0.5)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions         = model.predict(rescaledValidationX)
svm_accuracy=accuracy_score(Y_validation, predictions)*100


# ================== prepare MLP model =======================================
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = MLPClassifier(alpha=0.0001, activation="tanh", solver="adam", learning_rate="constant", hidden_layer_sizes=(5,))
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
mlp_accuracy=accuracy_score(Y_validation, predictions)*100


# ================== prepare KNN model =======================================
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
knn_accuracy=accuracy_score(Y_validation, predictions)*100

# ================== prepare CART model =======================================
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = DecisionTreeClassifier(max_features='log2', min_samples_split=11, random_state=123, min_samples_leaf=5)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
cart_accuracy=accuracy_score(Y_validation, predictions)*100

# ================== prepare LDA model =======================================
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = LinearDiscriminantAnalysis(shrinkage=None, solver='lsqr')
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
lda_accuracy=accuracy_score(Y_validation, predictions)*100

# ================== prepare AB model =======================================
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
ab_accuracy=accuracy_score(Y_validation, predictions)*100

# ================== prepare GBM model =======================================
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingClassifier(n_estimators=250)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
gbm_accuracy=accuracy_score(Y_validation, predictions)*100

# ================== prepare RF model =======================================
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = RandomForestClassifier(n_jobs=-1, min_samples_leaf=1, n_estimators=20, random_state=123, criterion='gini', min_samples_split=5)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
rf_accuracy=accuracy_score(Y_validation, predictions)*100

# ================== prepare ET model =======================================
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = ExtraTreesClassifier(n_jobs=-1, min_samples_leaf=1, n_estimators=20, random_state=123, criterion='gini', min_samples_split=3)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
et_accuracy=accuracy_score(Y_validation, predictions)*100

print("\n-------------------- Accuracies after parameters tuning --------------------") 
print("\nSVM Accuracy: %s" % round(svm_accuracy,2))
print("\nMLP Accuracy: %s" % round(mlp_accuracy,2))
print("\nKNN Accuracy: %s" % round(knn_accuracy,2))
print("\nCART Accuracy: %s" % round(cart_accuracy,2))
print("\nLDA Accuracy: %s" % round(lda_accuracy,2))
print("\nAB Accuracy: %s" % round(ab_accuracy,2))
print("\nGBM Accuracy: %s" % round(gbm_accuracy,2))
print("\nRF Accuracy: %s" % round(rf_accuracy,2))
print("\nET Accuracy: %s" % round(et_accuracy,2))