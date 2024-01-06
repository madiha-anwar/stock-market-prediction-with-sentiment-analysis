# -*- coding: utf-8 -*-
"""

@author: Madiha
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:56:08 2019

@author: Madiha
"""

# ======================= Stock prediction using news  ======================

# ========================= load libraries ===================================
import pandas
import numpy

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

from pandas import set_option
from matplotlib import pyplot

from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = '10'
#rcParams['font.sans-serif'] = ['Tahoma']

from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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

# ========================= load dataset =====================================
#dataset_file = "datasets/Spams reduction/NYSE/10-micro-2.0y.csv"
#names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend', 'SocialSentiment', 'FutureTrend']

dataset_file = "datasets/Price data/MSFT/9-micro-2.0y.csv"
names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend', 'FutureTrend']


dataset = pandas.read_csv(dataset_file, names=names, skiprows=[0])
print("\nDataset: %s" % dataset_file)

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

# ========================= data types =======================================

#print("\n==================== data types ====================")
#set_option('display.max_rows', 500)
#print(dataset.dtypes)

#print("\n==================== statistical summary of dataset ====================")
#set_option('display.width', 100)
#set_option('precision', 3)
#description = dataset.describe()
#print(description)

print("\n==================== class distribution ====================")
class_counts = dataset.groupby('FutureTrend').size()
print(class_counts)

#print("\n==================== pearson correlations ====================")
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

# ========================= Split-out validation dataset =====================
array = dataset.values
X = array[:,0:7]
Y = array[:,7]
validation_size = 0.30
num_folds = 10
seed = 7
scoring = 'accuracy'


X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
test_size=validation_size, random_state=seed)

# ========================= Spot-Check Algorithms ============================
models = []
models.append(('GNB',GaussianNB()))
models.append(('MNB',MultinomialNB()))
models.append(('SVM', SVC()))
models.append(('LR', LogisticRegression()))
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
print("\n==================== GNB Accuracy on validation dataset ====================")
gnb_accuracy =[]
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
gnb_predictions = gnb.predict(X_validation)
gnb_accuracy = accuracy_score(Y_validation, gnb_predictions)*100
prediction_accuracy.append(gnb_accuracy)

print("Accuracy: %f" % gnb_accuracy)
print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, gnb_predictions))
print("\nClassification report:")
print(classification_report(Y_validation, gnb_predictions))

print("\n==================== MNB Accuracy on validation dataset ====================")
mnb_accuracy =[]
mnb = MultinomialNB()
mnb.fit(X_train, Y_train)
mnb_predictions = mnb.predict(X_validation)
mnb_accuracy = accuracy_score(Y_validation, mnb_predictions)*100
prediction_accuracy.append(mnb_accuracy)

print("Accuracy: %f" % mnb_accuracy)
print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, mnb_predictions))
print("\nClassification report:")
print(classification_report(Y_validation, mnb_predictions))

print("\n==================== SVM Accuracy on validation dataset ====================")
svm_accuracy =[]
svc = SVC()
svc.fit(X_train, Y_train)
svc_predictions = svc.predict(X_validation)
svm_accuracy = accuracy_score(Y_validation, svc_predictions)*100
prediction_accuracy.append(svm_accuracy)

print("Accuracy: %f" % svm_accuracy)
print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, svc_predictions))
print("\nClassification report:")
print(classification_report(Y_validation, svc_predictions))

print("\n==================== LR Accuracy on validation dataset ====================")
lr_accuracy =[]
lr = LogisticRegression()
lr.fit(X_train, Y_train)
lr_predictions = lr.predict(X_validation)
lr_accuracy = accuracy_score(Y_validation, lr_predictions)*100
prediction_accuracy.append(lr_accuracy)

print("Accuracy: %f" % lr_accuracy)
print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, lr_predictions))
print("\nClassification report:")
print(classification_report(Y_validation, lr_predictions))

print("\n==================== MLP Accuracy on validation dataset ====================")
mlp_accuracy =[]
mlp = MLPClassifier()
mlp.fit(X_train, Y_train)
mlp_predictions = mlp.predict(X_validation)
mlp_accuracy = accuracy_score(Y_validation, mlp_predictions)*100
prediction_accuracy.append(mlp_accuracy)

print("Accuracy: %f" % mlp_accuracy)
print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, mlp_predictions))
print("\nClassification report:")
print(classification_report(Y_validation, mlp_predictions))

print("\n==================== KNN Accuracy on validation dataset ====================")
knn_accuracy =[]
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
knn_predictions = knn.predict(X_validation)
knn_accuracy = accuracy_score(Y_validation, knn_predictions)*100
prediction_accuracy.append(knn_accuracy)

print("Accuracy: %f" % knn_accuracy)
print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, knn_predictions))
print("\nClassification report:")
print(classification_report(Y_validation, knn_predictions))

print("\n==================== CART Accuracy on validation dataset ====================")
cart_accuracy =[]
cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
cart_predictions = cart.predict(X_validation)
cart_accuracy = accuracy_score(Y_validation, cart_predictions)*100
prediction_accuracy.append(cart_accuracy)

print("Accuracy: %f" % cart_accuracy)
print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, cart_predictions))
print("\nClassification report:")
print(classification_report(Y_validation, cart_predictions))

print("\n==================== LDA Accuracy on validation dataset ====================")
lda_accuracy =[]
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
lda_predictions = lda.predict(X_validation)
lda_accuracy = accuracy_score(Y_validation, lda_predictions)*100
prediction_accuracy.append(lda_accuracy)

print("Accuracy: %f" % lda_accuracy)
print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, lda_predictions))
print("\nClassification report:")
print(classification_report(Y_validation, lda_predictions))

print("\n==================== AB Accuracy on validation dataset ====================")
ab_accuracy =[]
ab = AdaBoostClassifier()
ab.fit(X_train, Y_train)
ab_predictions = ab.predict(X_validation)
ab_accuracy = accuracy_score(Y_validation, ab_predictions)*100
prediction_accuracy.append(ab_accuracy)

print("Accuracy: %f" % ab_accuracy)
print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, ab_predictions))
print("\nClassification report:")
print(classification_report(Y_validation, ab_predictions))

print("\n==================== GBM Accuracy on validation dataset ====================")
gbm_accuracy =[]
gbm = GradientBoostingClassifier()
gbm.fit(X_train, Y_train)
gbm_predictions = gbm.predict(X_validation)
gbm_accuracy = accuracy_score(Y_validation, gbm_predictions)*100
prediction_accuracy.append(gbm_accuracy)

print("Accuracy: %f" % gbm_accuracy)
print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, gbm_predictions))
print("\nClassification report:")
print(classification_report(Y_validation, gbm_predictions))

print("\n==================== RF Accuracy on validation dataset ====================")
rf_accuracy =[]
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
rf_predictions = rf.predict(X_validation)
rf_accuracy = accuracy_score(Y_validation, rf_predictions)*100
prediction_accuracy.append(rf_accuracy)

print("Accuracy: %f" % rf_accuracy)
print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, rf_predictions))
print("\nClassification report:")
print(classification_report(Y_validation, rf_predictions))

print("\n==================== ET Accuracy on validation dataset ====================")
et_accuracy =[]
et = ExtraTreesClassifier()
et.fit(X_train, Y_train)
et_predictions = et.predict(X_validation)
et_accuracy = accuracy_score(Y_validation, et_predictions)*100
prediction_accuracy.append(et_accuracy)

print("Accuracy: %f" % et_accuracy)
print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, et_predictions))
print("\nClassification report:")
print(classification_report(Y_validation, et_predictions))

# ========================= Compare Algorithms graphically ===================

print("\n==================== compare models graphically with validation dataset ====================")

pyplot.bar(names, prediction_accuracy, color="LightSeaGreen")
#pyplot.xticks(y_pos, names)
pyplot.ylabel('Accuracy')
pyplot.title('Classifiers prediction accuracy')
 
pyplot.show()


# ========================= Standardize the dataset ==========================
print("\n==================== Evaluation results on standardized dataset ====================")
pipelines = []
pipelines.append(('ScaledGNB', Pipeline([('Scaler', StandardScaler()),('GNB',
GaussianNB())])))
    
pipelines.append(('ScaledMNB', Pipeline([('Scaler', MinMaxScaler()),('MNB',
MultinomialNB())])))
    
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM',
SVC())])))
    
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LogisticRegression())])))
    
pipelines.append(('ScaledMLP', Pipeline([('Scaler', StandardScaler()),('MLP',
MLPClassifier())])))
    
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsClassifier())])))
    
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeClassifier())])))
    
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))

pipelines.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB',
AdaBoostClassifier())])))
    
pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM',
GradientBoostingClassifier())])))
    
pipelines.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF',
RandomForestClassifier())])))

pipelines.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesClassifier())])))

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
# ================== prepare the models =======================================
# ================== prepare GNB model =======================================
print("\n==================== GNB Accuracy on transformed validation dataset ====================")
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GaussianNB()
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
gnb_accuracy=accuracy_score(Y_validation, predictions)*100
print("Accuracy: %s" % gnb_accuracy)
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))

# ================== prepare MNB model =======================================
print("\n==================== MNB Accuracy on transformed validation dataset ====================")
scaler = MinMaxScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = MultinomialNB()
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
mnb_accuracy=accuracy_score(Y_validation, predictions)*100
print("Accuracy: %s" % mnb_accuracy)
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))

# ================== prepare SVM model =======================================
print("\n==================== SVM Accuracy on transformed validation dataset ====================")
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = SVC(kernel="rbf", C=0.5)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
svm_accuracy=accuracy_score(Y_validation, predictions)*100
print("Accuracy: %s" % svm_accuracy)
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))

# ================== prepare LR model =======================================
print("\n==================== LR Accuracy on transformed validation dataset ====================")
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = LogisticRegression()
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
lr_accuracy=accuracy_score(Y_validation, predictions)*100
print("Accuracy: %s" % lr_accuracy)
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))

# ================== prepare MLP model =======================================
print("\n==================== MLP Accuracy on transformed validation dataset ====================")
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = MLPClassifier(alpha=0.0001, activation="tanh", solver="adam", learning_rate="constant", hidden_layer_sizes=(5,))
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
mlp_accuracy=accuracy_score(Y_validation, predictions)*100
print("Accuracy: %s" % mlp_accuracy)
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))

# ================== prepare KNN model =======================================
print("\n==================== KNN Accuracy on transformed validation dataset ====================")
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
knn_accuracy=accuracy_score(Y_validation, predictions)*100
print("Accuracy: %s" % knn_accuracy)
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))

# ================== prepare CART model =======================================
print("\n==================== CART Accuracy on transformed validation dataset ====================")
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = DecisionTreeClassifier(max_features='log2', min_samples_split=13, random_state=123, min_samples_leaf=1)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
cart_accuracy=accuracy_score(Y_validation, predictions)*100
print("Accuracy: %s" % cart_accuracy)
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))

# ================== prepare LDA model =======================================
print("\n==================== LDA Accuracy on transformed validation dataset ====================")
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = LinearDiscriminantAnalysis(shrinkage=None, solver='lsqr')
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
lda_accuracy=accuracy_score(Y_validation, predictions)*100
print("Accuracy: %s" % lda_accuracy)
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))

# ================== prepare AB model =======================================
print("\n==================== AB Accuracy on transformed validation dataset ====================")
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = AdaBoostClassifier(n_estimators=50, learning_rate=.05)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
ab_accuracy=accuracy_score(Y_validation, predictions)*100
print("Accuracy: %s" % ab_accuracy)
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))

# ================== prepare GBM model =======================================
print("\n==================== GBM Accuracy on transformed validation dataset ====================")
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingClassifier(n_estimators=400)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
gbm_accuracy=accuracy_score(Y_validation, predictions)*100
print("Accuracy: %s" % gbm_accuracy)
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))

# ================== prepare RF model =======================================
print("\n==================== RF Accuracy on transformed validation dataset ====================")
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = RandomForestClassifier(n_jobs=-1, min_samples_leaf=1, n_estimators=20, random_state=123, criterion='entropy', min_samples_split=6)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
rf_accuracy=accuracy_score(Y_validation, predictions)*100
print("Accuracy: %s" % rf_accuracy)
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))

# ================== prepare ET model =======================================
print("\n==================== ET Accuracy on transformed validation dataset ====================")
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = ExtraTreesClassifier(n_jobs=-1, min_samples_leaf=1, n_estimators=15, random_state=123, criterion='gini', min_samples_split=3)
model.fit(rescaledX, Y_train)
# ================= transform the validation dataset =========================
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
et_accuracy=accuracy_score(Y_validation, predictions)*100
print("Accuracy: %s" % et_accuracy)
print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
print("Classification report: %s" % classification_report(Y_validation, predictions))

print("\n==================== Accuracies ====================")
print("GNB: %s" % round(gnb_accuracy,2))
print("MNB: %s" % round(mnb_accuracy,2))
print("SVM: %s" % round(svm_accuracy,2))
print("LR: %s" % round(lr_accuracy,2))
print("MLP: %s" % round(mlp_accuracy,2))
print("KNN: %s" % round(knn_accuracy,2))
print("CART: %s" % round(cart_accuracy,2))
print("LDA: %s" % round(lda_accuracy,2))
print("AB: %s" % round(ab_accuracy,2))
print("GBM: %s" % round(gbm_accuracy,2))
print("RF: %s" % round(rf_accuracy,2))
print("ET: %s" % round(et_accuracy,2))