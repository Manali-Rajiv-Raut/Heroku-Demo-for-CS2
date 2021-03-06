# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:54:05 2021

@author: Manali
"""
from module_2_preprocessing import Data_Preprocessing
from module_12_DF_creation import DataFrame_Creation

#import joblib
import pandas as pd
import numpy as np
import pickle
import nltk
nltk.download('punkt')
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from flask import Flask, request, jsonify, render_template
from flask_table import Table, Col
from flask import Markup
import re

app = Flask(__name__)
path_acbsa = "saved_model/acbsa_model.h5"
path_sentiment = "saved_model/sentiment_model.h5"
path_tokenizer = 'saved_model/tokenizer'
path_le_acbsa = 'saved_model/label_encoder_acbsa'
path_le_sentiment = 'saved_model/label_encoder_sentiment'
#print(IPython.__version__)

# loading the tokenizer
with open(path_tokenizer, 'rb') as f:               
    tokenizer = pickle.load(f)

#Loading Lable encoder for aspect based sentiemnt analysis
with open(path_le_acbsa, 'rb') as f:               
    label_encoder_acbsa = pickle.load(f)

#Loading Lable encoder for sentiemnt analysis
with open(path_le_sentiment, 'rb') as f:               
    label_encoder_sentiment = pickle.load(f) 
    
# aspect catagory based analysis
def acbsa_model_creation():
    acbsa_model = Sequential(name="acbsa")                                                   
    acbsa_model.add(Dense(512, input_shape=(6000,), activation='relu'))
    acbsa_model.add((Dense(256, activation='relu')))
    acbsa_model.add((Dropout(0.3)))
    acbsa_model.add((Dense(128, activation='relu')))
    acbsa_model.add(Dense(5, activation='softmax'))
    acbsa_model.load_weights(path_acbsa)
    return acbsa_model

# sentiment analysis
def sentiment_model_creation():
    sentiment_model = Sequential(name="sentiment")
    sentiment_model.add(Dense(512, input_shape=(6000,), activation='relu'))
    sentiment_model.add((Dense(256, activation='relu')))
    sentiment_model.add((Dropout(0.3)))
    sentiment_model.add((Dense(128, activation='relu')))
    sentiment_model.add(Dense(4, activation='softmax'))
    sentiment_model.load_weights(path_sentiment)
    return sentiment_model
    
#loading the preprocessing class
dp = Data_Preprocessing()
#loading the darframe creation class
dfc = DataFrame_Creation()
# loading acbsa model
acbsa_model = acbsa_model_creation()
# loading sentiment model
sentiment_model = sentiment_model_creation()
     
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # store the given text in a variable
    text = request.form.get("text")
    sentence = nltk.tokenize.sent_tokenize(text)   
    sen_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(sentence))
    predicted_cat = label_encoder_acbsa.inverse_transform(np.argmax(acbsa_model.predict(sen_tokenized), axis=-1))       
    predicted_polarity =label_encoder_sentiment.inverse_transform(np.argmax(sentiment_model.predict(sen_tokenized), axis=-1))
    result = dfc.create_result_dataframe(predicted_cat,predicted_polarity)
    html = result.to_html() 
    return render_template('index.html', prediction_text= Markup(html))
    

if __name__ == "__main__":
    app.run(debug=True) 
    
    
