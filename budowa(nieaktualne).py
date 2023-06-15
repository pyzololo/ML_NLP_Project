# -*- coding: utf-8 -*-

#%% imports

import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import pickle
from datetime import datetime, timedelta
from dateutil import parser, tz
import pytz
from langdetect import detect
# pip install deep-translator
from deep_translator import GoogleTranslator
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import copy
from nltk.corpus import stopwords
import spacy
from sklearn.cluster import KMeans

#%% 
##### PREPROCESSING #####

#%% reading data

def read_data():
    
    import os

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
    
#%% parsing files to gain the data

data = read_data()

#%% normalize date including timezone

def normalize_datetime(date):
    if not date:  # Sprawdzenie, czy podany argument jest pusty
        return None
    
    match = re.search(r"[A-Z]{3}", date)
    date_obj = None
    
    tzinfos = {"IDT": tz.tzoffset(None, 3*60*60),
               "ACST": tz.tzoffset(None, 9*60*60),
               "KST": tz.tzoffset(None, 9*60*60),
               "UT": tz.tzoffset(None, 0),
               "NZST": tz.tzoffset(None, 12*60*60),
               "ECT": tz.tzoffset(None, 1*60*60),
               "PST": tz.tzoffset(None, -8*60*60),
               "CDT": tz.tzoffset(None, -5*60*60),
               "MST": tz.tzoffset(None, -7*60*60),
               "BST": tz.tzoffset(None, 1*60*60),
               "EDT": tz.tzoffset(None, -4*60*60),
               "CST": tz.tzoffset(None, -6*60*60),
               "EST": tz.tzoffset(None, -5*60*60),
               "MDT": tz.tzoffset(None, -6*60*60),
               "CET": tz.tzoffset(None, 1*60*60),
               "PDT": tz.tzoffset(None, -7*60*60),
               "MET": tz.tzoffset(None, 1*60*60),
               "MEZ": tz.tzoffset(None, 1*60*60),
               "TUR": tz.tzoffset(None, 3*60*60)}
    
    if match:
        try:
            date_obj = datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %Z")
            # date_obj = parser.parse(date, fuzzy=True, tzinfos=tzinfos)
        except ValueError:
            try:
                date_obj = parser.parse(date, fuzzy=True, tzinfos=tzinfos)
            except parser.ParserError:
                raise ValueError("Unknown date format: %s" % date)
    else:
        try:
            date_obj = parser.parse(date, fuzzy=True, tzinfos=tzinfos)
        except parser.ParserError:
            raise ValueError("Unknown date format: %s" % date)
    
    # Przetwarzanie daty bez użycia strefy czasowej
    normalized_date = date_obj.replace(tzinfo=None)
    
    timezone = pytz.timezone("Europe/London")  # Wybierz odpowiednią strefę czasową
    normalized_date = timezone.localize(normalized_date)
    
    return normalized_date

#%% just normalize date format, does not change timezone

def normalize_date(date):
    if not date:  # Check if the input argument is empty
        return None
    
    match = re.search(r"[A-Z]{3}", date)
    date_obj = None

    if match:
        try:
            date_obj = datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %Z")
        except ValueError:
            try:
                date_obj = parser.parse(date, fuzzy=True)
            except parser.ParserError:
                raise ValueError("Unknown date format: %s" % date)
    else:
        try:
            date_obj = parser.parse(date, fuzzy=True)
        except parser.ParserError:
            raise ValueError("Unknown date format: %s" % date)
    
    return date_obj

#%% adding NormalizedDate column in order to sort data before splitting train-test

data['NormalizedDate'] = data['Date'].apply(lambda date: normalize_datetime(date))
data['Date'] = data['Date'].apply(lambda date: normalize_date(date))

#%% extract day, month, hour to new columns

# we dont use neither year nor month, because the distribution is very skewed

data['NormalizedDate'] = pd.to_datetime(data['NormalizedDate'])

data['Day'] = data['NormalizedDate'].dt.day
data['Hour'] = data['NormalizedDate'].dt.hour
data['Minute'] = data['NormalizedDate'].dt.minute

#%% sorting data by NormalizedDate

df_sorted = data.loc[np.argsort(data['NormalizedDate']), :].reset_index()

#%% languages

def detect_language(t):
    try:
        return detect(t)            
    except:
        return 'other'
    
df_sorted['Language'] = df_sorted['Text'].apply(lambda t: detect_language(t))

# non_english_map = {57:'fr', 882: 'arabskie jakies', 1472: 'dutch', 1475: 'dutch',
#                    10840: 'jakis szwedzki', 12998: 'szwedzki', 12764: 'Sweden',
#                    15704: 'de'}

non_english_idx = [57, 882, 1472, 1475, 10840, 12998, 12764, 15704]

english_text_idx = list(set(df_sorted.index.tolist()).difference(set(non_english_idx)))

# changing incorrectly detected languages to english
df_sorted.loc[english_text_idx, 'Language'] = 'en'

#%% translation

df_sorted.loc[non_english_idx, 'Text'] = df_sorted.loc[non_english_idx, :].apply(lambda x: GoogleTranslator(source=x.Language, target='en').translate(x.Text), axis=1)

#%% splitting data

train_test_df, valid_df = train_test_split(df_sorted, test_size=0.3, shuffle=False)
train_df, test_df = train_test_split(train_test_df, test_size=0.3, shuffle=False)

# removing index column
train_df.drop('index', axis=1, inplace=True)
valid_df.drop('index', axis=1, inplace=True)
test_df.drop('index', axis=1, inplace=True)

#%% word tokenization

# extracting texts into variables
texts_train = copy.deepcopy(train_df['Text'])
texts_test = copy.deepcopy(test_df['Text'])
texts_valid = copy.deepcopy(valid_df['Text'])

# tokenizing texts
texts_train_tokenized = [word_tokenize(text.lower()) for text in texts_train]
texts_test_tokenized = [word_tokenize(text.lower()) for text in texts_test]
texts_valid_tokenized = [word_tokenize(text.lower()) for text in texts_valid]

#%% getting only alphabetic marks

alpha_texts_tokenized_train = [[word for word in text if word.isalpha()] for text in texts_train_tokenized]
alpha_texts_tokenized_test = [[word for word in text if word.isalpha()] for text in texts_test_tokenized]
alpha_texts_tokenized_valid = [[word for word in text if word.isalpha()] for text in texts_valid_tokenized]

#%% removing stop words

texts_tokenized_without_stopwords_train = [[word for word in text if word not in stopwords.words('english')] for text in alpha_texts_tokenized_train]
texts_tokenized_without_stopwords_test = [[word for word in text if word not in stopwords.words('english')] for text in alpha_texts_tokenized_test]
texts_tokenized_without_stopwords_valid = [[word for word in text if word not in stopwords.words('english')] for text in alpha_texts_tokenized_valid]

#%% lemmatizing

nlp = spacy.load('en_core_web_sm')

def lemmatize_words(doc):
    doc = nlp(" ".join(doc))
    return [token.lemma_ for token in doc]

texts_lemmatized_spacy_train = list(map(lemmatize_words, texts_tokenized_without_stopwords_train))
texts_lemmatized_spacy_test = list(map(lemmatize_words, texts_tokenized_without_stopwords_test))
texts_lemmatized_spacy_valid = list(map(lemmatize_words, texts_tokenized_without_stopwords_valid))

#%% tfidf vectorization

from sklearn.feature_extraction.text import TfidfVectorizer

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

# or this (depending on version)
# X_train_tfidf_df = pd.DataFrame(X_train_tfidf.todense().A, columns=tfidf_vectorizer.get_feature_names())
# X_valid_tfidf_df = pd.DataFrame(X_valid_tfidf.todense().A, columns=tfidf_vectorizer.get_feature_names())
# X_test_tfidf_df = pd.DataFrame(X_test_tfidf.todense().A, columns=tfidf_vectorizer.get_feature_names())

#%% 
##### MODEL #####

#%% detecting number of clusters

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

def metrics_plots(X, max_k=10):
    score = []
    score_kmeans_s = []
    score_kmeans_c = []
    score_kmeans_d = []

    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, random_state=101)
        predictions = kmeans.fit_predict(X)
        # Calculate cluster validation metrics and append to lists of metrics
        score.append(kmeans.score(X) * (-1)) # there was a mistake here before
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
        
metrics_plots(X_train_tfidf_df, 15)
# possible number of clusters: 4, 6, 9, 10, 11, 12, 14

#%% feature importance

def plot_feature_importance(X, y, limit = 10):
    # creating and fitting model
    decisionTreeClassifier = DecisionTreeClassifier(random_state=1)
    decisionTreeClassifier.fit(X, y)
    importance = decisionTreeClassifier.feature_importances_
    
    names = X.columns
    
    # creating arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    # creating a DataFrame using a Dictionary
    data={'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)
    
    # sorting the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    # selecting a subset of features
    fi_df = fi_df.iloc[:limit]
    
    # defining size of bar plot
    plt.figure(figsize=(10,8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title('FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    
#%% KMeans

# selected number of clusters = 12
n_clusters = 12

kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)

kmeans.fit(X_train_tfidf_df)

train_preds = kmeans.predict(X_train_tfidf_df)
test_preds = kmeans.predict(X_test_tfidf_df)

cluster_centers = kmeans.cluster_centers_

# we can add also a barplot to visualise devision
train_preds_summarized = pd.Series(train_preds).value_counts()
test_preds_summarized = pd.Series(test_preds).value_counts()

#%% feature importance

train_preds = np.array(train_preds)
plot_feature_importance(X_train_tfidf_df, train_preds, 20)

#%% draw barplot of cluster size - function

def plot_cluster_size_from_series(series, title):
    plt.bar(series.index, series.values)
    plt.xlabel('Values')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()
    
#%% KMeans - cluster sizes

plot_cluster_size_from_series(train_preds_summarized, 'Klastry KMeans zbiór treningowy')
plot_cluster_size_from_series(test_preds_summarized, 'Klastry KMeans zbiór testowy')

#%% print cluster contents - function

def print_cluster_contents(df, preds):
    '''
    for each cluster function summarizes how many of each directory end up inside

    Parameters
    ----------
    df : pd.DataFrame
        train_df, test_df or valid_df
    preds : numpy.ndarray
        the result of model prediction
    '''
    n = len(np.unique(preds))
    directories = df['Directory']

    for i in range(n):
        print(f"CLUSTER NO. {i}")
        indexes = np.where(preds == i)
        directories_inside = directories.iloc[indexes]
        print(directories_inside.value_counts())
        print()
        
#%% some insight into result
# checking which directories end up in each cluster

print_cluster_contents(train_df, train_preds)
print_cluster_contents(test_df, test_preds)

# bad: 9, 11, 13, 15, 16
# ok: 10, 12, 14
# best: 12

#%% other tested models
# ...




