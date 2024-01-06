# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:56:08 2019

@author: Madiha
"""

# ======================= Stocks prediction hardness  ======================

# ========================= load libraries ===================================
import pandas

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
dataset_file = "datasets/Stocks prediction hardness/HPQ_IBM_MSFT_ORCL_RHT_TWTR_MSI_NOK-ClosePrices-2.0y.csv"
names = ['Date', 'HPQ', 'IBM', 'MSFT', 'ORCL', 'RHT', 'TWTR', 'MSI', 'NOK']
dataset = pandas.read_csv(dataset_file, names=names, skiprows=[0])
print("\nDataset: %s" % dataset_file)

print("\n==================== Fluctuations in Closing Prices of different stocks ====================")

ax=dataset.plot(x='Date', label='HPQ')
ax.set_ylabel("Stocks Closing Price ($)")
 