# -*- coding: utf-8 -*-
"""

@author: Madiha
"""

# ======================= Stock prediction using LSTM  ======================

# ========================= load libraries ===================================
import pandas
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pandas import set_option
from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = '10'
#rcParams['font.sans-serif'] = ['Tahoma']

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from pandas import DataFrame
from numpy import array

from tensorflow.keras.callbacks import CSVLogger
import os
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm_notebook


def get_pandas_column_names(file_path):
  import pandas
  dataframe = pandas.read_csv(file_path)
  return dataframe.columns.values.tolist()


class LSTM:
  def __init__(self) -> None:
    pass

  @classmethod
  def run(self, dataset_file_path: str):
    # ========================= load dataset =====================================
    names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend', 'SocialSentiment', 'FutureTrend']
    names = get_pandas_column_names(dataset_file_path)
    dataset = pandas.read_csv(dataset_file_path, names=names, skiprows=[0])
    print("\nDataset: %s" % dataset_file_path)

    # ========================= count the number of NaN values in each column ====
    print("\n==================== number of NaN values in each column ====================")
    print(dataset.isnull().sum())
    # remove null values
    dataset.dropna(inplace=True)

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
    train_cols = ["Open","High","Low","Close","Volume", "Trend", "Twitter Sentiments","FutureTrend"]
    names.pop(0)       # removing the date column
    train_cols = names
    df_train, df_test = train_test_split(dataset, train_size=0.7, test_size=0.3, shuffle=False)
    print("Train and Test size", len(df_train), len(df_test))

    #================= params =======================
    params = {
        "batch_size": 32,  # 20<16<10, 25 was a bust
        "epochs": 100,
        "lr": 0.00010000,
        "time_steps": 503
    }

    iter_changes = "dropout_layers_0.4_0.4"
    path = "LSTM/"

    #INPUT_PATH = path+"/inputs"
    OUTPUT_PATH = path+"/outputs/lstm/"+iter_changes
    TIME_STEPS = params["time_steps"]
    BATCH_SIZE = params["batch_size"]
    EPOCHS = params["epochs"] 

    # scale the feature MinMax, build array
    x= df_train.loc[:,train_cols].values
    min_max_scaler= MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x)
    x_test = min_max_scaler.transform(df_test.loc[:,train_cols])

    def build_timeseries(mat, y_col_index):
        # y_col_index is the index of column that would act as output column
        # total number of time-series samples would be len(mat) - TIME_STEPS
        dim_0 = mat.shape[0] - TIME_STEPS
        dim_1 = mat.shape[1]
        x = np.zeros((dim_0, TIME_STEPS, dim_1))
        y = np.zeros((dim_0,))
        
        for i in (range(dim_0)):
            x[i] = mat[i:TIME_STEPS+i]
            y[i] = mat[TIME_STEPS+i, y_col_index]
        print("length of time-series i/o",x.shape,y.shape)
        return x, y

    def trim_dataset(mat, batch_size):
        """
        trims dataset to a size that's divisible by BATCH_SIZE
        """
        no_of_rows_drop = mat.shape[0]%batch_size
        if(no_of_rows_drop > 0):
            return mat[:-no_of_rows_drop]
        else:
            return mat
        
    x_t, y_t = build_timeseries(x_train, len(train_cols) - 1)
    x_t=trim_dataset(x_t, BATCH_SIZE)
    y_t = trim_dataset(y_t, BATCH_SIZE)
    x_temp, y_temp = build_timeseries(x_test, len(train_cols) - 1)
    x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),32)
    y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),32)

    #creating the model
    lstm_model = Sequential()
    lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform'))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(20,activation='relu'))
    lstm_model.add(Dense(1,activation='sigmoid'))
    optimizer = optimizers.RMSprop(lr=params["lr"])
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    #train the model
    csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 'your_log_name' + '.log'), append=True)

    history = lstm_model.fit(x_t, y_t, epochs=EPOCHS, verbose=2, batch_size=BATCH_SIZE,
                        shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                        trim_dataset(y_val, BATCH_SIZE)), callbacks=[csv_logger])

    print(history.history['loss'])
    print(history.history['acc'])
