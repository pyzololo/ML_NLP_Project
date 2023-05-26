#%% imports

import pandas as pd
import numpy as np
import re
import os
import pickle
from datetime import datetime, timedelta
from dateutil import parser, tz
import pytz
from langdetect import detect
from deep_translator import GoogleTranslator
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import copy
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt


#%% reading data

def read_data():
    
    data = []
    path = '20_newsgroups'


    # Use os.walk() to iterate through directories and files
    for root, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file)) as f:
                lines = f.readlines()

                # Find the index of the first blank line
                first_blank_line_index = next((i for i, line in enumerate(lines) if line.strip() == ''), None)

                # Extract header content
                # content = ''.join(lines[:first_blank_line_index]) ###########
                content = ''.join(lines[:])

                # Use regular expressions to extract header information
                From = re.search(r"From: (.*)", content)
                Subject = re.search(r"Subject: (.*)", content)
                Organization = re.search(r"Organization: (.*)", content)
                Lines = re.search(r"Lines: (\d+)", content)
                Date = re.search(r"Date: (.*)", content)
                Directory = root[14:] # we remove '20_newsgroups\'
                File = file

                # Store the extracted information and the text after the first blank line in a dictionary
                record = {
                    'From': From.group(1) if From else '',
                    'Subject': Subject.group(1) if Subject else '',
                    'Organization': Organization.group(1) if Organization else '',
                    'Lines': int(Lines.group(1)) if Lines else 0,
                    'Date': Date.group(1) if Date else '',
                    'Directory': Directory,
                    'File': File,
                    'Text': ''.join(lines[first_blank_line_index + 1:]).strip()
                }

                # Append the dictionary to the data list
                data.append(record)

    # Create the DataFrame using the data list
    df = pd.DataFrame(data, columns=['From', 'Subject', 'Organization', 'Lines', 'Date', 'Directory', 'File', 'Text'])
    
    return df

#%% saving lists functions

def saveList2(myList, filename):
    with open(filename, 'wb') as file:
        pickle.dump(myList, file)
        
#%% loading lists functions

def loadList2(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

#%% loading lemmatized data

texts_lemmatized_spacy_train = loadList2("texts_lemmatized_spacy_train.pickle")
texts_lemmatized_spacy_test = loadList2("texts_lemmatized_spacy_test.pickle")
texts_lemmatized_spacy_valid = loadList2("texts_lemmatized_spacy_valid.pickle")

#%% tfidf vectorization

train_texts = [" ".join(text) for text in texts_lemmatized_spacy_train]
valid_texts = [" ".join(text) for text in texts_lemmatized_spacy_valid]
test_texts = [" ".join(text) for text in texts_lemmatized_spacy_test]

# Inicjalizacja TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Trenowanie vectorizera na danych treningowych
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)

# Transformacja danych walidacyjnych i testowych
X_valid_tfidf = tfidf_vectorizer.transform(valid_texts)
X_test_tfidf = tfidf_vectorizer.transform(test_texts)

#%% changing type of the result to data frame

X_train_tfidf_df = pd.DataFrame(X_train_tfidf.todense().A, columns=tfidf_vectorizer.get_feature_names_out())
X_valid_tfidf_df = pd.DataFrame(X_valid_tfidf.todense().A, columns=tfidf_vectorizer.get_feature_names_out())
X_test_tfidf_df = pd.DataFrame(X_test_tfidf.todense().A, columns=tfidf_vectorizer.get_feature_names_out())

#%% number of clusters

def metrics_plots(X, max_k=10):
    score = []
    score_kmeans_s = []
    score_kmeans_c = []
    score_kmeans_d = []

    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, random_state=101)
        predictions = kmeans.fit_predict(X)
        # Calculate cluster validation metrics and append to lists of metrics
        score.append(kmeans.score(X))
        score_kmeans_s.append(silhouette_score(X, kmeans.labels_, metric='euclidean'))
        score_kmeans_c.append(calinski_harabasz_score(X, kmeans.labels_))
        score_kmeans_d.append(davies_bouldin_score(X, predictions))

    list_scores = [score, score_kmeans_s, score_kmeans_c, score_kmeans_d] 
    # Elbow Method plot
    list_title = ['Within-cluster sum of squares', 'Silhouette Score', 'Calinski Harabasz', 'Davies Bouldin'] 
    for i in range(len(list_scores)):
        x_ticks = list(range(2, len(list_scores[i]) + 2))
        plt.plot(x_ticks, list_scores[i], 'bx-')
        plt.xlabel('k')
        plt.ylabel(list_title[i])
        plt.title('Optimal k')
        plt.show()
        
metrics_plots(X_train_tfidf_df, 10)