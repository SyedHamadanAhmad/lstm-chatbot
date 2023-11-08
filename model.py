import random
import string
import numpy as np
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer()
from data import responses
from preprocess_data import input_shape, total_words, output_length, x_train, y_train
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Flatten, LSTM

i=Input(shape=(input_shape, ))
x=Embedding(total_words+1, 10)(i)
x=LSTM(10, return_sequences=True)(x)
x=Flatten()(x)
x=Dense(output_length, activation='softmax')(x)

model=Model(i,x)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train=model.fit(x_train, y_train, epochs=200)
model.save("LSTM Model", train)



