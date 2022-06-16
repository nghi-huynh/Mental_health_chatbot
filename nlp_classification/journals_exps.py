import csv
import time
import pandas as pd
import os
import sys
import numpy as np
import io
import gensim
import nltk
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, matthews_corrcoef

dataset = pd.read_csv('data/dataset_merged_mh_journalling.csv', index_col=0)
#nltk.download('punkt')
#nltk.download('stopwords')

def pre_processing(df):
    df['text'] = df['title'] + ' ' + df['selftext']
    df.drop(columns=['title', 'selftext'], inplace=True)
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace('https:\S+|www.\S+', '', case=False)
    df['text'] = df['text'].str.replace(r'[^\x00-\x7F]+', '', regex=True)
    df = df.dropna()
    df['token_text'] = df['text'].apply(nltk.word_tokenize)
    stop_words = nltk.corpus.stopwords.words("english")
    df['token_text'] = df['token_text'].apply(lambda x: [item for item in x if item not in stop_words])
    df['token_text'] = df['token_text'].apply(lambda x: [item for item in x if re.match('[a-z]+', item)])
    df = df.dropna()

    return df

def train_tfidf(df):
    dat= pd.read_csv('data/dataset_merged_mh_journalling.csv', index_col=0)
    X = pre_processing(dat)
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(X['token_text'].astype(str))
    df_trans = Tfidf_vect.transform(df['token_text'].astype(str))
    return df_trans

def load_pickle_model():
    # and later you can load it
    with open('mh_svm.pkl', 'rb') as f:
        clf = pickle.load(f)
    return clf

labels = ['Anxiety', 'Body Dysmorphia', 'Depression', 'Eating Disorder', 'PTSD', 'Suicide', 'none']

dataset = pd.read_csv('data/dataset_journal_entries.csv', index_col=0)
df = pre_processing(dataset)
df_trans = train_tfidf(df)
svm_mh = load_pickle_model()

preds = svm_mh.predict_proba(df_trans)
pred_label = svm_mh.predict(df_trans)

for i in range(0, 50):
    print(dataset.iloc[i],df.iloc[i], preds[i], pred_label[i])
    print('\n')