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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


model = keras.models.load_model('nlp_classification/content/mh_LSTM_simple')

#dataset = pd.read_csv('data/dataset_merged_mh_journalling.csv', index_col=0)
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


def load_pickle_model():
    # and later you can load it
    with open('mh_svm.pkl', 'rb') as f:
        clf = pickle.load(f)
    return clf


labels = ['Anxiety', 'Body Dysmorphia', 'Depression', 'Eating Disorder', 'PTSD', 'Suicide', 'none']

dataset = pd.read_csv('data/dataset_journal_entries.csv', index_col=0)
df = pre_processing(dataset)

#model = keras.models.load_model('nlp_classification/content/mh_LSTM_simple')

X = df['token_text']
X = tf.convert_to_tensor(X.to_numpy())
pred_label = model.predict(X, batch_size=64)

for i in range(0, 50):
    print(dataset.iloc[i],df.iloc[i], pred_label[i])
    print('\n')