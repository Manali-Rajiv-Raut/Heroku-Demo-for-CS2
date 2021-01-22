# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:54:05 2021

@author: Manali
"""
from module_2_preprocessing import Data_Preprocessing
from module_12_DF_creation import DataFrame_Creation

import joblib
import pandas as pd
import numpy as np
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras import utils as np_utils
from tensorflow.keras.layers import Dropout

from IPython.display import HTML 

from flask import Flask, request, jsonify, render_template
from flask_table import Table, Col

app = Flask(__name__)
path_acbsa = "saved_model/acbsa_model.h5"
path_sentiment = "saved_model/sentiment_model.h5"
path_tokenizer = 'saved_model/tokenizer'
path_le_acbsa = 'saved_model/label_encoder_acbsa'
path_le_sentiment = 'saved_model/label_encoder_sentiment'

with open(path_tokenizer, 'rb') as f:               
    tokenizer = pickle.load(f)

with open(path_le_acbsa, 'rb') as f:               
    label_encoder_acbsa = pickle.load(f)

with open(path_le_sentiment, 'rb') as f:               
    label_encoder_sentiment = pickle.load(f) 
    
#html = "" "

def acbsa_model_creation():
    acbsa_model = Sequential()                                                   
    acbsa_model.add(Dense(512, input_shape=(6000,), activation='relu'))
    acbsa_model.add((Dense(256, activation='relu')))
    acbsa_model.add((Dropout(0.3)))
    acbsa_model.add((Dense(128, activation='relu')))
    acbsa_model.add(Dense(5, activation='softmax'))
    acbsa_model.load_weights(path_acbsa)
    return acbsa_model

def sentiment_model_creation():
    sentiment_model = Sequential()
    sentiment_model.add(Dense(512, input_shape=(6000,), activation='relu'))
    sentiment_model.add((Dense(256, activation='relu')))
    sentiment_model.add((Dropout(0.3)))
    sentiment_model.add((Dense(128, activation='relu')))
    sentiment_model.add(Dense(4, activation='softmax'))
    sentiment_model.load_weights(path_sentiment)
    return sentiment_model 
  
    
@app.route('/')
def home():
    return render_template('index.html')


def html_results(df):
    html = df.to_html() 
    # write html to file 
    text_file = open("templates/result.html", "w") 
    text_file.write(html) 
    text_file.close() 


@app.route('/predict',methods=['POST'])
def predict():
    acbsa_model = acbsa_model_creation()
    sentiment_model = sentiment_model_creation()
    dp = Data_Preprocessing()
    dfc = DataFrame_Creation()
    
    # store the given text in a variable
    text = request.form.get("text")
    # split the text to get each line in a list
    text2 = text.split('\n')
    # change the text (add 'Hi' to each new line)
    sentence = [ line for line in text2]
    sen_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(sentence))
    predicted_cat = label_encoder_acbsa.inverse_transform(acbsa_model.predict_classes(sen_tokenized))        
    predicted_polarity = label_encoder_sentiment.inverse_transform(sentiment_model.predict_classes(sen_tokenized))
    result = dfc.create_result_dataframe(predicted_cat,predicted_polarity)
    
    table = html_results(result)
    #table.border = True
    return render_template('result.html', table=table)
    #return render_template('index.html', prediction_text='The rsult of ABSA is  {} '.format(result))

    

if __name__ == "__main__":
    app.run(debug=True)  
    
    
    
    
    
    
    
            