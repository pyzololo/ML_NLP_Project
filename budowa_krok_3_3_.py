#%% imports

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import os
import pickle
from datetime import datetime
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
from sklearn_extra.cluster import KMedoids
from sklearn import cluster
from sklearn import mixture

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

#%% saving lists function
    
def saveList2(myList, filename):
    with open(filename, 'wb') as file:
        pickle.dump(myList, file)
        
#%% loading lists functions

def loadList2(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

#%% parsing files to gain the data

df_raw = read_data()

#%% insight into data

type(df_raw)
df_raw.shape
df_raw.columns

for col in df_raw.columns:
    print("Column name: " + col)
    print(df_raw[col].head())
    print()

#%% normalize date including timezone

def normalize_datetime(date):
    if not date:  
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
    
    normalized_date = date_obj.replace(tzinfo=None)
    
    timezone = pytz.timezone("Europe/London") 
    normalized_date = timezone.localize(normalized_date)
    
    return normalized_date

#%% just normalize date format, does not change timezone

def normalize_date(date):
    if not date:
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

df_raw['NormalizedDate'] = df_raw['Date'].apply(lambda date: normalize_datetime(date))
df_raw['Date'] = df_raw['Date'].apply(lambda date: normalize_date(date))

#%% checking if the hour is different 

for i in range(len(df_raw['Date'])):
    hour1 = pd.to_datetime(df_raw['NormalizedDate'].loc[i]).to_pydatetime()
    hour2 = df_raw['Date'].loc[i]
    if hour1.hour != hour2.hour:
        print('The hour is different')

#%% extract day, month, hour to new columns
# we dont use year neither month, because the distribution is very skewed

df_raw['NormalizedDate'] = pd.to_datetime(df_raw['NormalizedDate'])

df_raw['Day'] = df_raw['NormalizedDate'].dt.day
df_raw['Hour'] = df_raw['NormalizedDate'].dt.hour
df_raw['Minute'] = df_raw['NormalizedDate'].dt.minute

#%% sorting data by NormalizedDate

df_sorted = df_raw.loc[np.argsort(df_raw['NormalizedDate']), :].reset_index()

#%% languages

def detect_language(t):
    try:
        return detect(t)            
    except:
        return 'other'
    
df_sorted['Language'] = df_sorted['Text'].apply(lambda t: detect_language(t))

non_english_idx = [57, 882, 1472, 1475, 10840, 12998, 12764, 15704]

english_text_idx = list(set(df_sorted.index.tolist()).difference(set(non_english_idx)))

# changing incorrectly detected languages to english
df_sorted.loc[english_text_idx, 'Language'] = 'en'

#%% translation

df_sorted.loc[non_english_idx, 'Text'] = df_sorted.loc[non_english_idx, :].apply(lambda x: GoogleTranslator(source=x.Language, target='en').translate(x.Text), axis=1)

print(df_sorted.loc[non_english_idx[0], 'Text'])
print(df_sorted.loc[non_english_idx[0], 'Language'])

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

#%% extracting extra information (not used)
# before the numbers and special marks are removed, it would be nice to get some statistics from them ...
# it should be done in a different way

def count_punctuation_marks(tokens, marks):
    count = 0
    def count_punctuation_marks_in_token(token, marks):
        count = 0
        for l in token:
            if l in marks:
                count += 1
        return count
    for token in tokens:
        count += count_punctuation_marks_in_token(token, marks)
    return count

def count_all_marks(tokens):
    count = 0
    for token in tokens:
        count += len(token)
    return count

def get_all_marks_statistics(texts):
    # summarizing punctuation marks
    dots = [count_punctuation_marks(tokens, ["."]) for tokens in texts]
    commas = [count_punctuation_marks(tokens, [","]) for tokens in texts]
    qms = [count_punctuation_marks(tokens, ["?"]) for tokens in texts]
    exs = [count_punctuation_marks(tokens, ["!"]) for tokens in texts]
    # summarizing other marks
    characters = ["<", ">", "/", "`", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "+", "=", ";", ":", "'", "{", "}", "[", "]", "|", "\"", '"']
    oths = [count_punctuation_marks(tokens, characters) for tokens in texts]
    # summarizing digits
    digits = [count_punctuation_marks(tokens, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]) for tokens in texts]
    # counting words in texts
    token_words = [len(tokens) for tokens in texts]
    # summarizing all marks
    all_marks = np.array([count_all_marks(tokens) for tokens in texts])
    # counting frequency
    dots = np.array(dots) / all_marks
    np.nan_to_num(dots, copy=False, nan=0.0) # works inplace
    commas = np.array(commas) / all_marks
    np.nan_to_num(commas, copy=False, nan=0.0)
    qms = np.array(qms) / all_marks
    np.nan_to_num(qms, copy=False, nan=0.0)
    exs = np.array(exs) / all_marks
    np.nan_to_num(exs, copy=False, nan=0.0)
    oths = np.array(oths) / all_marks
    np.nan_to_num(oths, copy=False, nan=0.0)
    digits = np.array(digits) / all_marks
    np.nan_to_num(digits, copy=False, nan=0.0)
    token_words = np.array(token_words)
    # adding average length of words in text
    words_len = all_marks / token_words
    np.nan_to_num(words_len, copy=False, nan=0.0)
    # creating a data frame from created columns
    df = pd.DataFrame({"dots":dots, "commas":commas, "qms":qms, 
                       "exs":exs, "oths":oths, "digits":digits, 
                       "token_words":token_words, "all_marks": all_marks, 
                       "words_len":words_len})
    
    return df
    
# summarizing punctuation marks
df_additional_train = get_all_marks_statistics(texts_train_tokenized)
df_additional_test = get_all_marks_statistics(texts_test_tokenized)
df_additional_valid = get_all_marks_statistics(texts_valid_tokenized)

#%% standarization (not used)

scaler = StandardScaler()
df_normalized_train = pd.DataFrame(scaler.fit_transform(df_additional_train), columns = df_additional_train.columns)
df_normalized_test = pd.DataFrame(scaler.fit_transform(df_additional_test), columns = df_additional_test.columns)
df_normalized_valid = pd.DataFrame(scaler.fit_transform(df_additional_valid), columns = df_additional_valid.columns)

#%% getting only alphabetic marks

alpha_texts_tokenized_train = [[word for word in text if word.isalpha()] for text in texts_train_tokenized]
alpha_texts_tokenized_test = [[word for word in text if word.isalpha()] for text in texts_test_tokenized]
alpha_texts_tokenized_valid = [[word for word in text if word.isalpha()] for text in texts_valid_tokenized]

#%% removing stop words

texts_tokenized_without_stopwords_train = [[word for word in text if word not in stopwords.words('english')] for text in alpha_texts_tokenized_train]
texts_tokenized_without_stopwords_test = [[word for word in text if word not in stopwords.words('english')] for text in alpha_texts_tokenized_test]
texts_tokenized_without_stopwords_valid = [[word for word in text if word not in stopwords.words('english')] for text in alpha_texts_tokenized_valid]

#%% saving data without stopwords and punctuation

saveList2(texts_tokenized_without_stopwords_train, "texts_tokenized_without_stopwords_train.pickle")
saveList2(texts_tokenized_without_stopwords_test, "texts_tokenized_without_stopwords_test.pickle")
saveList2(texts_tokenized_without_stopwords_valid, "texts_tokenized_without_stopwords_valid.pickle")

#%% loading data without stopwords and punctuation
    
texts_tokenized_without_stopwords_train = loadList2("texts_tokenized_without_stopwords_train.pickle")
texts_tokenized_without_stopwords_test = loadList2("texts_tokenized_without_stopwords_test.pickle")
texts_tokenized_without_stopwords_valid = loadList2("texts_tokenized_without_stopwords_valid.pickle")

#%% lemmatizing

nlp = spacy.load('en_core_web_sm')

def lemmatize_words(doc):
    doc = nlp(" ".join(doc))
    return [token.lemma_ for token in doc]

texts_lemmatized_spacy_train = list(map(lemmatize_words, texts_tokenized_without_stopwords_train))
texts_lemmatized_spacy_test = list(map(lemmatize_words, texts_tokenized_without_stopwords_test))
texts_lemmatized_spacy_valid = list(map(lemmatize_words, texts_tokenized_without_stopwords_valid))

#%% saving lemmatized data

saveList2(texts_lemmatized_spacy_train, "texts_lemmatized_spacy_train.pickle")
saveList2(texts_lemmatized_spacy_test, "texts_lemmatized_spacy_test.pickle")
saveList2(texts_lemmatized_spacy_valid, "texts_lemmatized_spacy_valid.pickle")

#%% loading lemmatized data

texts_lemmatized_spacy_train = loadList2("texts_lemmatized_spacy_train.pickle")
texts_lemmatized_spacy_test = loadList2("texts_lemmatized_spacy_test.pickle")
texts_lemmatized_spacy_valid = loadList2("texts_lemmatized_spacy_valid.pickle")

#%% tfidf vectorization

train_texts = [" ".join(text) for text in texts_lemmatized_spacy_train]
valid_texts = [" ".join(text) for text in texts_lemmatized_spacy_valid]
test_texts = [" ".join(text) for text in texts_lemmatized_spacy_test]

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)

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

#%% adding punctuation marks statistics (not used)

# X_train_tfidf_df = pd.concat([X_train_tfidf_df, df_normalized_train.reindex(X_train_tfidf_df.index)], axis=1)
# X_test_tfidf_df = pd.concat([X_test_tfidf_df, df_normalized_test.reindex(X_test_tfidf_df.index)], axis=1)
# X_valid_tfidf_df = pd.concat([X_valid_tfidf_df, df_normalized_valid.reindex(X_valid_tfidf_df.index)], axis=1)

#%% merging data together (not used)

# # for now only X_train_tfidf_df can be used, but some additional columns may be added after some operations (mapping, standarization)
# chosen_columns = ['From', 'Subject', 'Organization', 'Lines', 'Day', 'Hour', 'Minute', 'Language']
# X_train_tfidf_df = pd.concat([train_df[chosen_columns], X_train_tfidf_df], axis=1)

#%% detecting number of clusters

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

def make_prediction(X, y):
    decisionTreeClassifier = DecisionTreeClassifier(random_state=1)
    decisionTreeClassifier.fit(X, y)
    y_pred = decisionTreeClassifier.predict(X)
    return y_pred
    
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
    
#%% clustering scores (from labs) (not used - too much time consuming)

def count_clustering_scores(X, cluster_num, model, score_fun):
    if isinstance(cluster_num, int):
        cluster_num_iter = [cluster_num]
    else:
        cluster_num_iter = cluster_num
        
    scores = []    
    for k in cluster_num_iter:
        model_instance = model(n_clusters=k)
        labels = model_instance.fit_predict(X)
        wcss = score_fun(X, labels)
        scores.append(wcss)
    
    if isinstance(cluster_num, int):
        return scores[0]
    else:
        return scores
    
from scipy.spatial import distance

def _change_type_to_numpy(X):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    return X

def min_interclust_dist(X, label):
    clusters = set(label)
    global_min_dist = np.inf
    for cluster_i in clusters:
        cluster_i_idx = np.where(label == cluster_i)
        for cluster_j in clusters:
            if cluster_i != cluster_j:
                cluster_j_idx = np.where(label == cluster_j)
                interclust_min_dist = np.min(distance.cdist(X[cluster_i_idx], X[cluster_j_idx]))
                global_min_dist = np.min([global_min_dist, interclust_min_dist])
    return global_min_dist

def _inclust_mean_dists(X, label):
    clusters = set(label)
    inclust_dist_list = []
    for cluster_i in clusters:
        cluster_i_idx = np.where(label == cluster_i)
        inclust_dist = np.mean(distance.pdist(X[cluster_i_idx]))
        inclust_dist_list.append(inclust_dist)
    return inclust_dist_list

def mean_inclust_dist(X, label):
    inclust_dist_list = _inclust_mean_dists(X, label)
    return np.mean(inclust_dist_list)

def std_dev_of_inclust_dist(X, label):
    inclust_dist_list = _inclust_mean_dists(X, label)
    return np.std(inclust_dist_list)

def mean_dist_to_center(X, label):
    clusters = set(label)
    inclust_dist_list = []
    for cluster_i in clusters:
        cluster_i_idx = np.where(label == cluster_i)
        cluster_i_mean = np.mean(X[cluster_i_idx], axis=0, keepdims=True)
        inclust_dist = np.mean(distance.cdist(X[cluster_i_idx], cluster_i_mean))
        inclust_dist_list.append(inclust_dist)
    return np.mean(inclust_dist_list)

def summary(X, label):
    X = _change_type_to_numpy(X)
    results = []
    results.append(min_interclust_dist(X, label))
    results.append(mean_inclust_dist(X, label))
    results.append(std_dev_of_inclust_dist(X, label))
    results.append(mean_dist_to_center(X, label))
    labels = ['min dist between clusters', 'mean dist in clust', 'std dev dist in clust', 'mean dist to clust center']
    results = [results]
    S = pd.DataFrame(data = results, columns = labels)
    return S

#%% KMeans
# final model

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

y = np.array(train_preds)
plot_feature_importance(X_train_tfidf_df, y, 20)

#%% draw barplot of cluster size - function

def plot_cluster_size_from_series(series, title):
    plt.bar(series.index, series.values)
    plt.xlabel('Nr klastra')
    plt.ylabel('Liczność')
    plt.title(title)
    plt.show()
    
#%% KMeans - cluster sizes

plot_cluster_size_from_series(train_preds_summarized, 'Klastry KMeans zbiór treningowy')
plot_cluster_size_from_series(test_preds_summarized, 'Klastry KMeans zbiór testowy')

# similar results for train and test

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
        
        for index, count in directories_inside.value_counts().head(3).items():
            print(index, count)
        print()

#%% some insight into result
# checking which directories end up in each cluster

print_cluster_contents(train_df, train_preds)
print_cluster_contents(test_df, test_preds)

# examining number of clusters:
# bad: 9, 11, 13, 15, 16
# ok: 10, 12, 14
# best: 12

#%% check specific text form specific cluster (for train set)

# indexes = np.where(train_preds == 5) # choose cluster
# temp = train_df.iloc[indexes]
# my_directories = temp['Directory']
# my_texts = temp['Text']

# my_index = 223 # choose text
# print('\n'.join(['\t' + line for line in my_texts.iloc[my_index].split('\n')]))
# print(my_directories.iloc[my_index])

#%% get top keywords from each cluster

cluster_labels = kmeans.labels_

data = test_texts

cluster_names = {}

for i in range(12):
    # Get the data points belonging to the current cluster
    cluster_data = [data[j] for j in range(len(data)) if cluster_labels[j] == i]
    
    # Extract keywords from the cluster data using TF-IDF
    tfidf_vector = tfidf_vectorizer.transform(cluster_data)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_keywords = tfidf_vector.toarray().argsort()[:, ::-1][:, :50]  # Get top 3 keywords
    
    # Generate cluster name using top keywords
    cluster_name = ', '.join([feature_names[keyword_index] for keyword_index in top_keywords[0]])
    cluster_names[i] = cluster_name
    
for cluster_label, cluster_name in cluster_names.items():
    print(f"Cluster {cluster_label}: {cluster_name}")

#%% K-Medoids

kmedoids = KMedoids(n_clusters=9, random_state=0)
kmedoids.fit(X_train_tfidf_df)
y_kmedoids_train = kmedoids.predict(X_train_tfidf_df)
y_kmedoids_test = kmedoids.predict(X_test_tfidf_df)
centers = kmedoids.cluster_centers_

train_preds_summarized_medoids = pd.Series(y_kmedoids_train).value_counts()
test_preds_summarized_medoids = pd.Series(y_kmedoids_test).value_counts()

# everything in one cluster, idk why

#%% K-Medoids - cluster size

plot_cluster_size_from_series(train_preds_summarized_medoids, 'Klastry K-Medoids zbiór treningowy')
plot_cluster_size_from_series(test_preds_summarized_medoids, 'Klastry K-Medoids zbiór testowy')

#%% Mini Batch (unused)

miniBatchKmeans = cluster.MiniBatchKMeans(n_clusters=9, random_state=0)
miniBatchKmeans.fit(X_train_tfidf_df)
y_mbatch_train = miniBatchKmeans.predict(X_train_tfidf_df)
y_mbatch_test = miniBatchKmeans.predict(X_test_tfidf_df)
centers = miniBatchKmeans.cluster_centers_

train_preds_summarized_mbatch = pd.Series(y_mbatch_train).value_counts()
test_preds_summarized_mbatch = pd.Series(y_mbatch_test).value_counts()

#%% Mini Batch - cluster size (unused)

plot_cluster_size_from_series(train_preds_summarized_mbatch, 'Klastry Mini Batch zbiór treningowy')
plot_cluster_size_from_series(test_preds_summarized_mbatch, 'Klastry Mini Batch zbiór testowy')

#%% printing cluster contents Mini BAtch (unused)

print_cluster_contents(train_df, y_mbatch_train)
print_cluster_contents(test_df, y_mbatch_test)

#%% DBSCAN (unused)

# here the eps value should be optimized
dbs = cluster.DBSCAN(n_jobs=-1)
dbs.fit(X_train_tfidf_df)
dbs.labels_

#%% GMM (unused)

gmm = mixture.GaussianMixture(n_components=8, random_state=0)
gmm.fit(X_train_tfidf_df)
# problems with memory

#%% dendrogram (unused) - too much data

# from scipy.cluster.hierarchy import linkage, dendrogram

# # method parameter should be checked
# Z = linkage(X_train_tfidf_df, method='single')

# saveList2(Z, "Z.pickle")

# Z = loadList2("Z.pickle")

# # plt.figure(figsize=(10, 5), dpi= 200, facecolor='w', edgecolor='k')
# #plt.figure(facecolor='w', edgecolor='k')

# # this line restarts the kernel
# dendrogram(Z)
# plt.show()

#%% PCA (unused)

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import plotly.express as px

# pca = PCA(n_components=3)
# X_train = ...
# reduced_features = pca.fit_transform(X_train.toarray())

# df_pca = pd.DataFrame(data = reduced_features, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

# # plt.scatter(reduced_features[:,0], reduced_features[:,1], c=train_preds)
# fig = px.scatter_3d(reduced_features[:,0], reduced_features[:,1], reduced_features[:,2])

# reduced_cluster_centers = pca.transform(cluster_centers)

# # plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
# # plt.show()

# fig = px.scatter_3d(reduced_features)
# fig.show()

