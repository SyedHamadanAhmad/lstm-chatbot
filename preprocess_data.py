from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from data import data
tokenizer=Tokenizer()

#tokenizing data
tokenizer.fit_on_texts(data['Inputs'])
train=tokenizer.texts_to_sequences(data['Inputs'])

#padding
x_train=pad_sequences(train, padding='pre')

#encode tags
label_encoder=LabelEncoder()
y_train=label_encoder.fit_transform(data['Tags'])

total_words=len(tokenizer.word_index)
output_length=label_encoder.classes_.shape[0]

print(output_length)