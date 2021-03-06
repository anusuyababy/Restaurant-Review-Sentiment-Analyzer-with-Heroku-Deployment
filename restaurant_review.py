# -*- coding: utf-8 -*-
"""restaurant review

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xD-jZJhUSKgf86kPG4-RAh3vxy2e44qC
"""

import pickle
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import re

df=pd.read_csv('review.tsv', sep='\t')
df.head()

df.shape

df.isnull().sum()

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range (0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()

pickle.dump(cv, open('tfidf.pkl', 'wb'))

#y=df['Liked']
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('accuracy_score: ', accuracy_score(y_test, y_pred))

pickle.dump(clf, open('model.pkl', 'wb'))