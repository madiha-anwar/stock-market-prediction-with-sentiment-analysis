# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:56:08 2019

@author: Madiha
"""

# ======================= Stocks relationships  ======================

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


#from sklearn.neighbors import NearestNeighbors


# ========================= load dataset =====================================
dataset_file = "datasets/Stocks relationships/KSE_LSE_NYSE-Trends-2.0y.csv"
names = ['KSE', 'LSE', 'NYSE']
dataset = pandas.read_csv(dataset_file, names=names, skiprows=[0])
print("\nDataset: %s" % dataset_file)


# ========================= handle missing values ============================
#print("\n==================== number of missing values in each column ====================")
#print((dataset[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend','NewsSentiment','FutureTrend']] == 'NA').sum())
#dataset[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend','NewsSentiment','FutureTrend']] = dataset[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend','NewsSentiment','FutureTrend']].replace('NA', 0)

#dataset[['NewsSentiment']] = dataset[['NewsSentiment']].replace({'0':numpy.NaN, 0:numpy.NaN})
#dataset.dropna(inplace=True)

#print((dataset[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend','NewsSentiment','FutureTrend']] == 0).sum())
# ========================= count the number of NaN values in each column ====
#print("\n==================== number of NaN values in each column ====================")
#print(dataset.isnull().sum())
# remove null values
#dataset.dropna(inplace=True)

# ========================= print the first 5 rows of dataset ================
#print(dataset.head(5))
#print(dataset.tail(10))
#print(dataset[-50:])


print("\n==================== dimension of dataset ====================")
print(dataset.shape)

print("\n==================== histogram ====================")
dataset.hist()
pyplot.show()

print("\n==================== Pearson Correlations ====================")

correlations = dataset.corr(method='pearson').round(2)
print(correlations)

print("\n==================== Correlation Matrix ====================")
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = numpy.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()