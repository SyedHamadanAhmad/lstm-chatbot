from preprocess_data import input_shape, total_words, output_length, x_train, y_train
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Flatten, LSTM

i=Input(shape=(input_shape, ))
x=Embedding(total_words+1, 30,)(i)
x=LSTM(30, return_sequences=True)(x)
x=Flatten()(x)
x=Dense(output_length, activation='softmax')(x)

model=Model(i,x)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train=model.fit(x_train, y_train, epochs=200)
