# -*- coding: utf-8 -*-
"""

@author: Madiha
"""


# ======================= Stock prediction using LSTM  ======================
import pandas
import numpy as np
from numpy import argmax
from tensorflow.keras import optimizers

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import np_utils

#from imblearn.over_sampling import SMOTE
#import random

from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = '10'
#rcParams['font.sans-serif'] = ['Tahoma']

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM



from tqdm._tqdm_notebook import tqdm_notebook

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

# ======= load dataset =======
dataset_file = "datasets/News/IBM/9-micro-2.0y.csv"

dataset      = pandas.read_csv(dataset_file, usecols=[1,2,3,4,5,7,8])

print("\nDataset file: %s" % dataset_file)

# fix random seed for reproducibility (to get same results each time the code is executed)
np.random.seed(7)

#========================================================================================
# Instantiate encoder/scaler
open_price_scaler  = MinMaxScaler(feature_range = (0, 1))
high_price_scaler  = MinMaxScaler(feature_range = (0, 1))
low_price_scaler   = MinMaxScaler(feature_range = (0, 1))
close_price_scaler = MinMaxScaler(feature_range = (0, 1))
volume_price_scaler= MinMaxScaler(feature_range = (0, 1))
sentiment_scaler   = MinMaxScaler(feature_range = (0, 1))

# ============= encode categorical attribute values ========
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# Scale and Encode Separate Columns
dataset.iloc[:,0:1]  = open_price_scaler.fit_transform(dataset.iloc[:,0:1])
dataset.iloc[:,1:2]  = high_price_scaler.fit_transform(dataset.iloc[:,1:2])
dataset.iloc[:,2:3]  = low_price_scaler.fit_transform(dataset.iloc[:,2:3])
dataset.iloc[:,3:4]  = close_price_scaler.fit_transform(dataset.iloc[:,3:4])
dataset.iloc[:,4:5]  = volume_price_scaler.fit_transform(dataset.iloc[:,4:5])
dataset.iloc[:,5:6]  = sentiment_scaler.fit_transform(dataset.iloc[:,5:6])

dataset.iloc[:,-1]   = le.fit_transform(dataset.iloc[:,-1])

print("======= Number of Classes ========")
num_classes = len(list(set(dataset.iloc[:,-1])))
print(num_classes)


#dataset = dataset.astype('float32')
print("\n======= Dimension of dataset =========")
print(dataset.shape)

# ========================= Split-out dataset =====================
df_train, df_test = train_test_split(dataset, train_size=0.7, test_size=0.3, shuffle=False)



print("\n======= Train and Test size========\n", len(df_train), len(df_test))

#================= params =======================
params = {
    "batch_size": 32,  # 20<16<10, 25 was a bust
    "epochs": 300,
    "time_steps": 60,
    "n_features":7,
    "n_classes":3
}

TIME_STEPS   = params["time_steps"]
BATCH_SIZE   = params["batch_size"]
EPOCHS       = params["epochs"]
N_FEATURES   = params["n_features"]
N_CLASSES    = params["n_classes"]



x_train         = df_train.iloc[:,0:7].values
'''
sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train[:,0:3], x_train[:,-1])
#print(np.column_stack((x_train_res, y_train_res)))
'''
x_test          = df_test.iloc[:,0:7].values

#print(x_test)

def build_timeseries(mat, y_col_index):
    """
    Converts ndarray into timeseries format and supervised data format. Takes first TIME_STEPS
    number of rows as input and sets the TIME_STEPS+1th data as corresponding output and so on.
    :param mat: ndarray which holds the dataset
    :param y_col_index: index of column which acts as output
    :return: returns two ndarrays-- input and output in format suitable to feed
    to LSTM.
    """
    # total number of time-series samples would be len(mat) - TIME_STEPS
    
    #dim_0 is train_size-time steps i.e 351-10=341 for time step 10
    dim_0   = mat.shape[0] - TIME_STEPS
    
    #dim_1 is no of features i.e 4
    dim_1   = mat.shape[1]
    
    x       = np.zeros((dim_0, TIME_STEPS, dim_1))
    y       = np.zeros((dim_0,))
    
    for i in tqdm_notebook(range(dim_0)):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
        
    print("\n======= Length of time-series i/o ========\n",x.shape,y.shape)
    return x, y

def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE, i.e 32
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    
    if(no_of_rows_drop > 0):
        #returns total-dropped rows
        return mat[:-no_of_rows_drop]
    else:
        #returens total rows
        return mat
'''
x_train_total = np.column_stack((x_train_res, y_train_res))

random.shuffle(x_train_total)
print(x_train_total)
'''
x_t, y_t        =  build_timeseries(x_train, 6)

x_t             =  trim_dataset(x_t, BATCH_SIZE)

y_t             =  trim_dataset(y_t, BATCH_SIZE)

y_t_ohe         =  np_utils.to_categorical(y_t, 3)

#x_test reshaped
x_temp, y_temp  =  build_timeseries(x_test, 6)


#divides the array into two sub arrays
x_val, x_test_t =  np.split(trim_dataset(x_temp, BATCH_SIZE),2)

y_val, y_test_t =  np.split(trim_dataset(y_temp, BATCH_SIZE),2)

#one hot y_val
y_val_ohe       = np_utils.to_categorical(y_val, 3)

#one hot y_test_t
y_test_t_ohe    = np_utils.to_categorical(y_test_t, 3)

#creating the model
LSTM_Model      =  Sequential()
LSTM_Model.add(LSTM(100, return_sequences=True, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), 
                    dropout=0.2, recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform'))

LSTM_Model.add(LSTM(100, dropout=0.2, return_sequences=True))
LSTM_Model.add(LSTM(100, dropout=0.2))
LSTM_Model.add(Dense(N_FEATURES,activation='relu'))
LSTM_Model.add(Dense(N_CLASSES, activation='softmax'))
#in exceptional case, whn the number of classes are two (Positive and Negative)
#LSTM_Model.add(Dense(len(list(set(y_t))),activation='softmax'))

#set the learning rate
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#sgd = optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
LSTM_Model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

'''
class_weight = {0: 1.,
                1: 10.,
                2: 1.}
'''
history = LSTM_Model.fit(x_t, y_t_ohe, epochs=EPOCHS, verbose=2, batch_size=BATCH_SIZE,
                    shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                    trim_dataset(y_val_ohe, BATCH_SIZE)))


print("\n==================== Evaluate the model ====================")

loss, accuracy          = LSTM_Model.evaluate(x_t, y_t_ohe, verbose=0)

print('Training Loss: %f, Training Accuracy: %f' % (loss, accuracy*100))

val_loss, val_accuracy  = LSTM_Model.evaluate(x_val, y_val_ohe, verbose=0)
print('Validation Loss: %f, Validation Accuracy: %f' % (val_loss, val_accuracy*100))

#print("\n==================== Model Summary ====================")
#print(LSTM_Model.summary())

print("\n======== Make predictions =========")
#predictions = lstm_model.predict_classes(x_t, verbose=0)

y_pred  = LSTM_Model.predict_classes(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)

#y_pred  = y_pred.flatten()
print(y_pred)
y_pred_inv = le.inverse_transform(y_pred)
print(y_pred_inv)

y_test_t= trim_dataset(y_test_t_ohe, BATCH_SIZE)

print("\n======== Performance evaluation =========")
print("Confusion matrix: \n%s" % confusion_matrix(y_test_t.argmax(axis=1), y_pred))
print("Classification report: \n%s" % classification_report(y_test_t.argmax(axis=1), y_pred))
print("Cohen kappa score: \n%s" % cohen_kappa_score(y_test_t.argmax(axis=1), y_pred))

#============ Visualize the training data ===============
print("\n======== Model accuracy and loss plotting =========")
from matplotlib import pyplot as plt
plt.figure()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epochs')
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training Loss', 'Testing Loss'], loc='upper left')
plt.show()