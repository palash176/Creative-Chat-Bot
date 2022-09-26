import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import webbrowser
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("category.csv")


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    ps = PorterStemmer()
    text = ''.join([ps.stem(j) for j in text])
    return text


def process_text(text):
    df['description'] = df['description'].apply(clean_text)
    dff = pd.DataFrame()
    text = [text]
    dff['des'] = text
    text = dff['des'].apply(clean_text)
    return text



def model(text):
    cv = CountVectorizer(stop_words='english') 
    cv_matrix = cv.fit_transform(df['description'])
    X = pd.DataFrame(cv_matrix.toarray(), index=df['description'].values, columns=cv.get_feature_names())
    cv_matrix_test = cv.transform(text)
    X_test = pd.DataFrame(cv_matrix_test.toarray(), index=text.values, columns=cv.get_feature_names())
    
    dic = {'hello':1, 'buy':2, 'dship':3, 'work':4, 'demo':5, 'products':6, 'consume':7, 'bye':8, 'fine':9, 'intro':10}
    y = df['category'].map(dic)
    
    
    model2=GradientBoostingClassifier(n_estimators=30, max_depth=4, learning_rate=0.1)
    model2.fit(X, y)
    y_pred = model2.predict(X_test)
    
    rdic = {1:'hello', 2:'buy', 3:'dship', 4:'work',5:'demo', 6:'products', 7:'consume', 8:'bye', 9:'fine', 10:'intro'}
    y_pred = pd.DataFrame(y_pred)
    y_pr = y_pred[0].map(rdic)
    return y_pr


import random
import json
f = open("intents.json",)

data = json.load(f)


def make_response(text):
    text = process_text(text)
    category = model(text)
    category=category[0]
    for intent in data['intents']:
        tag=intent['tag']
        tag=str(tag)
        if (tag==category):
            return random.choice(intent['responses'])
        
from flask import Flask, render_template
from flask import request, redirect
from csv import writer
app = Flask(__name__)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/get')
def get_bot_response():
    message = request.args.get('msg')
    if message:
        text=message
        text=process_text(text)
        text=model(text)
        text=text[0]
        li=[message,text]
        with open('category.csv','a') as file:
            writer_object=writer(file)
            writer_object.writerow(li)
            #file.close()
        text_video(make_response(message))
        hologram()
        return make_response(message)
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0')     

