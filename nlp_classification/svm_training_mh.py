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
from sklearn.calibration import CalibratedClassifierCV
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


def train_SVM(df):
    # train SVM
    X = df['token_text'].astype(str)
    Y = df['class'].astype(str)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, random_state=42, test_size=0.2,
                                                                        stratify=Y)
    classes_names = [
        'Suicide',
        'Depression',
        'PTSD',
        'Anxiety',
        'Eating Disorder',
        'Body Dysmorphia'
    ]

    encoder = LabelEncoder()
    encoder.fit(classes_names)
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)
    print(encoder.classes_)
    integer_mapping = {l: i for i, l in enumerate(encoder.classes_)}
    print(integer_mapping)

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(X)
    Train_X_trans = Tfidf_vect.transform(X_train)
    Test_X_trans = Tfidf_vect.transform(X_test)

    vocab_dict = Tfidf_vect.vocabulary_
    labels = ['Anxiety', 'Body Dysmorphia', 'Depression', 'Eating Disorder', 'PTSD', 'Suicide', 'none']

    svm_model = svm.LinearSVC(random_state=42)
    svm_model = CalibratedClassifierCV(svm_model)
    svm_model.fit(Train_X_trans, y_train_enc)
    preds_SVM = svm_model.predict(Test_X_trans)
    print(svm_model.score(Test_X_trans, y_test_enc))
    print(svm_model.get_params())
    print("SVM Accuracy Score", accuracy_score(y_test_enc, preds_SVM))
    print("Confusion matrix:\n", sklearn.metrics.confusion_matrix(y_test_enc, preds_SVM))
    print("F1:", sklearn.metrics.f1_score(y_test_enc, preds_SVM, average='weighted'))
    print("Precision:", sklearn.metrics.precision_score(y_test_enc, preds_SVM, average='weighted'))
    print("Recall:", sklearn.metrics.recall_score(y_test_enc, preds_SVM, average='weighted'))
    print("Classification Report:\n", sklearn.metrics.classification_report(y_test_enc, preds_SVM, target_names=labels))

    return svm_model


def save_pickle_model(clf):
    with open('mh_svm.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print('pickle saved')
    return


def load_pickle_model():
    # and later you can load it
    with open('mh_svm.pkl', 'rb') as f:
        clf = pickle.load(f)
    return clf


# go through with training the svm, save model
df = pre_processing(dataset)
svm_mh = train_SVM(df)
save_pickle_model(svm_mh)
