import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from sklearn.metrics import roc_auc_score
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

uno_train = np.empty([0, 482])
uno_test = np.empty([0, 482])
for i in range(4):
    file = np.load('feature_new/train.%d.npy' %(i+1))
    uno_train = np.append(uno_train, file, axis=0)
for i in range(2):
    file = np.load('feature_new/test.%d.npy' %(i+1))
    uno_test = np.append(uno_test, file, axis=0)

bobo_train = np.load('svm/trainX.npy')
bobo_test = np.load('svm/testX.npy')

trainY = uno_train[:,0:2]
testY = uno_test[:,0:2]

trainX_380 = np.append(bobo_train, uno_train[:, -8:], axis=1)
trainX_521 = np.append(bobo_train, uno_train[:, 325:474], axis=1)
trainX_695 = np.append(bobo_train, uno_train[:, 2:325], axis=1)
testX_380 = np.append(bobo_test, uno_test[:, -8:], axis=1)
testX_521 = np.append(bobo_test, uno_test[:, 325:474], axis=1)
testX_695 = np.append(bobo_test, uno_test[:, 2:325], axis=1)

model = Sequential()
model.add(Dense(32, input_dim=380, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(trainX_380, trainY, batch_size=64, epochs=10, validation_split=0.2)

pred_y = model.predict_proba(testX_380)
print(roc_auc_score(testY[:,1], pred_y[:,1]))
