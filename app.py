from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle
import bz2
import pickle
import _pickle as cPickle

with open("news.pkl",'rb') as f:
    indices = pickle.load(f)


# Load any compressed pickle file
def decompress_pickle(file):
 data = bz2.BZ2File(file, 'rb')
 data = cPickle.load(data)
 return data
 
with open("vectorizer.pbz2",'rb') as fs:
    tfidf_matrix = decompress_pickle(fs)

data = pd.read_csv("News.csv")
data = data.head(9000)

indices = pd.Series(data.index, index=data['Title']).drop_duplicates()

app = Flask(__name__)
@app.route('/sa')
def check():
    return "Codegnan is in NBKR"
@app.route('/')
def satya():
    return render_template('index.html')

def news_recommendation(Title, similarity = tfidf_matrix):
    index = indices[Title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores,
    key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:10]
    newsindices = [i[0] for i in similarity_scores]
 
    recommendations = data.iloc[newsindices][['Title', 'News Category']]
    return recommendations
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['News']
    recommendations = news_recommendation(user_input)
    print(type(recommendations))
    if recommendations.empty:
        prediction = "No relevant news found."
    else:
        if len(recommendations):
            predit = pd.DataFrame(recommendations)
            pred = predit.get('Title','News Category')
            pds = [i for i in pred]
            prediction= pds
        else:
            prediction = "The News You searched is not relevant to any other news."
        
    return render_template('index.html',prediction=prediction )

app.run()