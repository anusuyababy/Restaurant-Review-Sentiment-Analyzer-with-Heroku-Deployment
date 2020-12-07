# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:30:21 2020

@author: DELL
"""
from flask import Flask, render_template, request
import pickle


# Load the Random Forest CLassifier model

clf = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('tfidf.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        data = [review]
        fact = cv.transform(data).toarray()
        my_prediction = clf.predict(fact)
        
        if(int(my_prediction)==1):
            prediction="WOO! THIS IS A POSITIVE REVIEW"
        else:
            prediction="IT IS A NEGATIVE REVIEW"
        
        return (render_template('index.html', prediction=prediction))


        
if __name__ == '__main__':
    app.run(debug=True)