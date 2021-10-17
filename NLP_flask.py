#!/usr/bin/env python
# coding: utf-8

# In[71]:


import nltk
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
toNumeric = CountVectorizer()
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
from flask import Flask, request, render_template


#app=FastAPI()

app = Flask("__name__")

q = ""


@app.route("/")
def loadPage():
	return render_template('home.html', query="")


class Item(BaseModel):
    text: str

toNumeric=pickle.load(open('tranform.pkl','rb'))



def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    #text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[^a-zA-Z]', ' ',text)
    return text

stop_words = stopwords.words('english')
more_stopwords = ['u', 'im', 'c']
stop_words = stop_words + more_stopwords

def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text
    

def porter_clean(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text


def Lan_stem(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

#@app.post('/test')
@app.route("/", methods=['POST'])
def finance_sentiment():
    #print(inputs)
    #inputs=inputs.dict()
    #text = inputs['text']
    text = request.form['query1']
    print(text)
    text = clean_text(text)
    text = remove_stopwords(text)
    text = porter_clean(text)
    text = Lan_stem(text)
    print(text)
    transformed_text = toNumeric.transform([text])
    print(transformed_text)
    rf = pickle.load(open('NLP_AWS_RF_picklev1.pkl', 'rb'))
    print(rf)
    pred_label = rf.predict(transformed_text)
    print(pred_label)
    print(type(pred_label))
    a = str(pred_label)
    print(type(a))
    print(text)
    if a=="[0]":
        print('Neutral')
        b = 'Neutral'
    elif a=="[1]":
        print('Positive')
        b =  'Positive'
    elif a=="[2]":
        print('Negative')
        b =  'Negative'
    
    return render_template('home.html', output1=b, query1 = request.form['query1'])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

# In[ ]:




