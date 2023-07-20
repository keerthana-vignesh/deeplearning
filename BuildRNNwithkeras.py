import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical

import warnings
warnings.filterwarnings('ignore')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

max_features = 1000
maxlen = 80
batch_size = 32

print('Loading data...')
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words = max_features)
print(len(x_train),'train sequences')
print(len(x_test),'test sequences')

print('pad sequences (samples x time)')
x_train = pad_sequences(x_train, maxlen = maxlen)
x_test = pad_sequences(x_test,maxlen = maxlen)
print('x_train shape', x_train.shape)
print('x_test shape', x_test.shape)

INDEX_FROM = 3
word_to_id = imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] =0 
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}
print(' '.join(id_to_word[id] for id in x_train[10]))

print("Build model.........................")
model = Sequential()
model.add(Embedding(max_features,8))
model.add(LSTM(16, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=batch_size,epochs=3,validation_data=(x_test,y_test))

score,acc = model.evaluate(x_test,y_test,batch_size=batch_size)
print('test score = ',score)
print('test accuracy = ',acc)

prediction = model.predict(x_test[220:221])
print("Prediction Value : ",prediction[0])
print("Original Value : ",y_test[220:221])
