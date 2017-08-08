# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 17:29:02 2017

@author: lakshya.khanna
"""

import os
import pandas as pd
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction import text
import numpy as np
from nltk import stem
from nltk import word_tokenize
from sklearn import tree
from sklearn import model_selection

os.chdir('E:\Kaggle\Bag of Words')

#Readinf Train Data
train_movie = pd.read_csv(filepath_or_buffer='E:\\Kaggle\\Bag of Words\\labeledTrainData.tsv\\labeledTrainData.tsv',delimiter = '\t',quoting = 3,header = 0)

#Reading Test Data
test_movie = pd.read_csv(filepath_or_buffer='E:\\Kaggle\\Bag of Words\\TestData.tsv\\TestData.tsv',delimiter = '\t',quoting = 3,header = 0)

train_movie.head(1)
train_movie.dtypes
train_movie.sentiment.unique()

#checking preprocessing on the data sample
review_tmp = train_movie['review'][0]
bs  = BeautifulSoup(review_tmp).getText()
review_tmp = re.sub('[^a-zA-Z]',' ',review_tmp)
review_tmp_words = review_tmp.split(' ')

sm = stem.PorterStemmer()
sm.stem('organization')
sm.stem('pruning')

ssm = stem.SnowballStemmer(language='english')
ssm.stem('organization')

lm = stem.WordNetLemmatizer()
lm.lemmatize('organization')
lm.lemmatize('modification')


def preprocess_review(review):
    review_text = BeautifulSoup(review).getText()
    review_text = re.sub('[^a-zA-Z]',' ',review_text)
    return review_text.lower()
    
def tokenize_rvw(review):
    return review.split(' ')

def lemmatizer_tokenizer(review):
    return [lm.lemmatize(token) for token in word_tokenize(review)]

vectorizer = text.CountVectorizer(preprocessor = preprocess_review, \
                                  ngram_range=(2, 2),  \
                                  tokenizer = tokenize_rvw,    \
                                  stop_words = 'english',   \
                                  max_features = 5000)


#transform the reviews to count vectors(dtm). Creates a DF containig all the words s features and their frequency in 
#in the reviews as the value of the feature. Same as that of DTM . The features/tokens/words are given number for the
#sake of having simple matrix.Check and open features in variable explorer.
features_df = vectorizer.fit_transform(train_movie.loc[0:4,'review']).toarray()

features = vectorizer.get_feature_names()
stopwords = vectorizer.get_stop_words()
ftr_index_no = vectorizer.vocabulary_

#check the distribution of features across reviews
dist = features_df.sum(axis=0)
for tag, count in zip(ftr_index_no, dist):
    print(count, tag)

##########
#Use lemmatizer as tokenizer and do fir all reviews
lemma_vectorizer = text.CountVectorizer(preprocessor = preprocess_review, \
                                  ngram_range=(2, 2),  \
                                  tokenizer = lemmatizer_tokenizer,    \
                                  stop_words = 'english'   \
                                  ,max_features = 5000
                                  )

lm_features_df = lemma_vectorizer.fit_transform(train_movie['review']).toarray()

lm_features = lemma_vectorizer.get_feature_names()
lm_stopwords = lemma_vectorizer.get_stop_words()
lm_ftr_index_no = lemma_vectorizer.vocabulary_

#check the distribution of lemma features across reviews
lm_dist = lm_features_df.sum(axis=0)
for tag, count in zip(lm_ftr_index_no, lm_dist):
    print(count, tag)

#Model building
X_train = lemma_vectorizer.fit_transform(train_movie['review']).toarray()
y_train = train_movie['sentiment']

dt_estimator = tree.DecisionTreeClassifier(random_state = 2017)
cv_score = model_selection.cross_val_score(dt_estimator, X_train, y_train, cv = 10 , n_jobs = 10)
dt_estimator.fit(X_train,y_train)

X_test = lemma_vectorizer.fit_transform(test_movie['review']).toarray()

test_movie['sentiment'] =  dt_estimator.predict(X_test)

test_movie.to_csv('bow_v1.csv',columns = ['id','sentiment'], index = False)

