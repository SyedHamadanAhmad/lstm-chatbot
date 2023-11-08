import json
import string
import pandas as pd

with open('intents.json', 'r') as content:
    data=json.load(content)
    

tags=[]
inputs=[]
responses={}

for intent in data['intents']:
    #creating a dictionary with a key value pair key-> tag, value->responses
    responses[intent['tag']]=intent['responses'] 
    for input in intent['patterns']:
        inputs.append(input)
        tags.append(intent['tag'])

data=pd.DataFrame({"Inputs":inputs, "Tags":tags})
data['Inputs']=data['Inputs'].apply(lambda word:[letter.lower() for letter in word if word not in string.punctuation])
data['Inputs']=data['Inputs'].apply([lambda word: ''.join(word)])

