# -*- coding: utf-8 -*-
"""

@author: Madiha
"""

# ======================= Dimensionality reduction/Features selection  ======================

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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


class Features_Selection_KBest:
  def __init__(self) -> None:
    pass

  def run(self, dataset_file_path: str):
    # ========================= load dataset =====================================
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

    print("\n==================== dimension of dataset ====================")
    print(dataset.shape)

    print("\n==================== class distribution ====================")
    class_counts = dataset.groupby('FutureTrend').size()
    print(class_counts)

    # ========================= Split-out validation dataset =====================
    array = dataset.values
    X = array[:,0:8]
    Y = array[:,8]


    X_new = SelectKBest(chi2, k=6).fit_transform(X, Y)


    validation_size = 0.30
    num_folds = 10
    seed = 7
    scoring = 'accuracy'


    X_train, X_validation, Y_train, Y_validation = train_test_split(X_new, Y,
    test_size=validation_size, random_state=seed)


    # ========================= Spot-Check Algorithms ============================
    models = []
    models.append(('GNB',GaussianNB()))

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

    print("\n==================== Evaluation results using SelectKBest ====================")

    # ================== prepare the models =======================================
    # ================== prepare GNB model =======================================
    print("\n==================== GNB Accuracy on transformed validation dataset ====================")
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)

    model = GaussianNB()

    # ================= transform the validation dataset =========================
    rescaledValidationX = scaler.transform(X_validation)


    model.fit(rescaledX, Y_train)
    predictions = model.predict(rescaledValidationX)



    gnb_accuracy=accuracy_score(Y_validation, predictions)*100
    print("Accuracy: %s" % gnb_accuracy)
    print("Confusion matrix: %s" % confusion_matrix(Y_validation, predictions))
    print("Classification report: %s" % classification_report(Y_validation, predictions))

    # ================== prepare MNB model =======================================
    print("\n==================== MNB Accuracy on transformed validation dataset ====================")
    scaler = MinMaxScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    #rescaledX_pca=pca.fit_transform(rescaledX)

    model = MultinomialNB()
    model.fit(rescaledX, Y_train)
    # ================= transform the validation dataset =========================
    rescaledValidationX = scaler.transform(X_validation)
    #rescaledValidationX_pca=pca.transform(rescaledValidationX)

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

    model = AdaBoostClassifier(n_estimators=50, learning_rate=0.05)
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

    model = RandomForestClassifier(n_jobs=-1, min_samples_leaf=1, n_estimators=10, random_state=123, criterion='entropy', min_samples_split=6)
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