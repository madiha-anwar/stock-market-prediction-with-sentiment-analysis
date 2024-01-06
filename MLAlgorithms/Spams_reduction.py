# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 09:04:35 2019

@author: Madiha
"""
# ======================= Tweets classification into ham and spam  ======================

# ========================= load libraries ===================================

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import nltk
nltk.download('stopwords')
from pandas import set_option


#from subprocess import check_output
#print(check_output(["datasets", "datasets/Spams reduction"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
names = ['Source', 'Text', 'Date', 'CashTags', 'Sentiment', 'Class']
trainingdata = pd.read_csv("datasets/Spams reduction/hamspamTweets.csv", encoding='latin-1', names=names, low_memory=False)

newdata = pd.read_csv("datasets/Spams reduction/HPQ_Raw_Tweets-2.0y.csv", encoding='latin-1', names=names, low_memory=False)

newdata['Class']=newdata['Class'].astype(str)
newdata[['Text']] = newdata[['Text']].replace({'':np.NaN, 0:np.NaN})
#newdata[['Sentiment']]=newdata[['Sentiment']].astype(object)
#newdata[['Text']] = newdata[['Text']].fillna({'Text':''})

#print("\n==================== number of NaN values in each column ====================")
#print(newdata.isnull().sum())
# remove null values
#dataset.dropna(inplace=True)
#newdata.fillna(0,inplace=True)

print(newdata.dtypes)

#data = data.drop(["Source", "Date", "Sentiment", "CashTags"], axis=1)
#newdata = newdata.drop(["Source", "Date", "Sentiment", "CashTags"], axis=1)
#data = data.rename(columns={"v1":"class", "v2":"text"})
newdata.head()

#data['length'] = data['Text'].apply(len)


def pre_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

#trainingdata['Text']=trainingdata['Text'].astype(str)
textFeatures = trainingdata['Text'].copy()
#textFeatures=textFeatures.astype(str)
textFeatures = textFeatures.apply(pre_process)
vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(textFeatures)

features_train, features_test, labels_train, labels_test = train_test_split(features, trainingdata['Class'], test_size=0.3, random_state=111)

from sklearn.naive_bayes import MultinomialNB

mnb_model = MultinomialNB(alpha=0.2)
mnb_model.fit(features_train, labels_train)
predictions = mnb_model.predict(features_test)
#print("prediction %s" % prediction)
classification_accuracy=accuracy_score(labels_test,predictions)
print("Accuracy: %s" % classification_accuracy)
print("Confusion matrix: %s" % confusion_matrix(labels_test, predictions))
print("Classification report: %s" % classification_report(labels_test, predictions))
#print("Predictions: %s" % prediction)

# new instances where we do not know the answer

newTextFeatures = newdata['Text'].copy()
#newTextFeatures =newTextFeatures.astype(str)

newTextFeatures = newTextFeatures.apply(pre_process)
#vectorizer = TfidfVectorizer("english")
newFeatures = vectorizer.transform(newTextFeatures)

# make a prediction

newclasses = mnb_model.predict(newFeatures)
# show the inputs and predicted outputs

predictedclass=[]
for i in range(len(newdata['Text'])):
    predictedclass.append(newclasses[i])

newdata['Class']=predictedclass
newdata.to_csv("datasets/Spams reduction/NYSE_Raw_Tweets_Classified-2.0y.csv", index=False)
