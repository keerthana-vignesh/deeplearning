import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings('ignore')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

max_features = 6000
maxlen = 80
batch_size = 32

data = pd.read_csv(r'C:\Users\odind\OneDrive\Desktop\python\rnn\SentiTest.csv')
data = data[['Sentiment','SentimentText']]

data['SentimentText'] = data['SentimentText'].apply(lambda x: x.lower())
data['SentimentText'] = data['SentimentText'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
print(data)
# for idx,row in data.iterrows():
#     row[0] = row[0].replace('rt',' ')
    
max_fatures = 2000
tokenizer = Tokenizer(nb_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['SentimentText'].values)
X = tokenizer.texts_to_sequences(data['SentimentText'].values)
X = pad_sequences(X)

Y = pd.get_dummies(data['Sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.50, random_state = 42)
print('Shape of training samples:',X_train.shape,Y_train.shape)
print('Shape of testing samples:',X_test.shape,Y_test.shape)

print("Build model.........................")

# MNB = MultinomialNB()
# MNB.fit(X_train, Y_train)
model = Sequential()
model.add(Embedding(2000,32))
model.add(LSTM(130,activation="tanh",recurrent_activation="hard_sigmoid"))
model.add(Dense(2,activation='softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
#batch_size = 32
model.fit(X_train, Y_train, epochs = 15, batch_size=max_features, verbose = 2,validation_data=(X_test,Y_test))

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = max_features)
print("Score: %.2f" % (score))
print("Accuracy: %.2f" % (acc))

while True:
    text = input()
    tester = np.array([text])
    tester = pd.DataFrame(tester)
    tester.columns = ['text']

    tester['text'] = tester['text'].apply(lambda x: x.lower())
    tester['text'] = tester['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

    max_fatures = 2000
    test = tokenizer.texts_to_sequences(tester['text'].values)
    test = pad_sequences(test)

    if X.shape[1]>test.shape[1]:
        test = np.pad(test[0], (X.shape[1]-test.shape[1],0), 'constant')
        
    test = np.array([test])
    prediction = model.predict(test)
    print(prediction[0],'prediction')
    # if prediction[0]<0.5:
    #     print('Negative')
    # else:
    #     print('Positive')