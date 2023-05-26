#%% imports

import pandas as pd
import numpy as np
import re

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

# this one is not needed anymore i guess
# def saveList(myList,filename):
#     # the filename should mention the extension 'npy'
#     np.save(filename,myList)
#     print("Saved successfully!")
    
import pickle

def saveList2(myList, filename):
    with open(filename, 'wb') as file:
        pickle.dump(myList, file)
        
#%% loading lists functions

# this one is not needed anymore i guess
# def loadList(filename):
#     # the filename should mention the extension 'npy'
#     tempNumpyArray=np.load(filename, allow_pickle=True)
#     return tempNumpyArray.tolist()

def loadList2(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

#%% parsing files to gain the data

# works a bit slower now, but still within 1 minute

gigatest = read_data()


#%% insight into data

type(gigatest)
gigatest.shape
gigatest.columns

for col in gigatest.columns:
    print("Column name: " + col)
    print(gigatest[col].head())
    print()

# in some texts there are still some attributes (Archive-name, Version, etc.), it may be removed
print(gigatest.loc[0,'Text'][0:1000])
print(gigatest.loc[1,'Text'][0:1000])
print(gigatest.loc[2000,'Text'][0:1000])


#%% normalize date including timezone

import re
from datetime import datetime, timedelta
from dateutil import parser, tz
import pytz

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

gigatest['NormalizedDate'] = gigatest['Date'].apply(lambda date: normalize_datetime(date))
gigatest['Date'] = gigatest['Date'].apply(lambda date: normalize_date(date))

#%% checking if the hour is different 

for i in range(len(gigatest['Date'])):
    hour1 = pd.to_datetime(gigatest['NormalizedDate'].loc[i]).to_pydatetime()
    hour2 = gigatest['Date'].loc[i]
    if hour1.hour != hour2.hour:
        print('The hour is different')

# it turns out, that the function that was supposed to include time zone in 
# datetime normalization is a piece of crap, it should be probably deleted, but
# there are more important tasks now

#%% extract day, month, hour to new columns

# we dont use year neither month, because the distribution is very skewed

gigatest['NormalizedDate'] = pd.to_datetime(gigatest['NormalizedDate'])

gigatest['Day'] = gigatest['NormalizedDate'].dt.day
# it does not take modified hours
gigatest['Hour'] = gigatest['NormalizedDate'].dt.hour
gigatest['Minute'] = gigatest['NormalizedDate'].dt.minute

# np.unique(gigatest['Day'].value_counts().index) # complete set
# len(np.unique(gigatest['Hour'].value_counts().index)) # complete
# len(np.unique(gigatest['Minute'].value_counts().index)) # complete

#%% sorting data by NormalizedDate

df_sorted = gigatest.loc[np.argsort(gigatest['NormalizedDate']), :].reset_index()


#%% languages

from langdetect import detect

def detect_language(t):
    try:
        return detect(t)            
    except:
        return 'other'
    
df_sorted['Language'] = df_sorted['Text'].apply(lambda t: detect_language(t))

non_english_map = {57:'fr', 882: 'arabskie jakies', 1472: 'dutch', 1475: 'dutch',
                   10840: 'jakis szwedzki', 12998: 'szwedzki', 12764: 'Sweden',
                   15704: 'de'}

non_english_idx = [57, 882, 1472, 1475, 10840, 12998, 12764, 15704]

english_text_idx = list(set(df_sorted.index.tolist()).difference(set(non_english_idx)))

# changing incorrectly detected languages to english
df_sorted.loc[english_text_idx, 'Language'] = 'en'

#%% translation?

#pip install deep-translator
from deep_translator import GoogleTranslator

# # example
# to_translate = 'I want to translate this text'
# translated = GoogleTranslator(source='auto', target='de').translate(to_translate)

df_sorted.loc[non_english_idx, 'Text'] = df_sorted.loc[non_english_idx, :].apply(lambda x: GoogleTranslator(source=x.Language, target='en').translate(x.Text), axis=1)

print(df_sorted.loc[non_english_idx[0], 'Text'])
print(df_sorted.loc[non_english_idx[0], 'Language'])


#%% splitting data

from sklearn.model_selection import train_test_split

train_test_df, valid_df = train_test_split(df_sorted, test_size=0.3, shuffle=False)
train_df, test_df = train_test_split(train_test_df, test_size=0.3, shuffle=False)

# train_test_df['Date']
# valid_df['Date']

#%% word tokenization

from nltk.tokenize import word_tokenize
import copy

# extracting texts into variables
texts_train = copy.deepcopy(train_df['Text'])
texts_test = copy.deepcopy(test_df['Text'])
texts_valid = copy.deepcopy(valid_df['Text'])

# # finds words, numbers and emails
# pattern = '([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$|\d+|\w+)'

# in both cases results are similar

# tokenizing texts
texts_train_tokenized = [word_tokenize(text.lower()) for text in texts_train]
texts_test_tokenized = [word_tokenize(text.lower()) for text in texts_test]
texts_valid_tokenized = [word_tokenize(text.lower()) for text in texts_valid]


#%% extracting extra information
# before the numbers and special marks are removed, it would be nice to get some statistics from them ...

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


#%% standarization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_normalized_train = scaler.fit_transform(df_additional_train)
df_normalized_test = scaler.transform(df_additional_test)
df_normalized_valid = scaler.transform(df_additional_valid)


#%% getting only alphabetic marks

alpha_texts_tokenized_train = [[word for word in text if word.isalpha()] for text in texts_train_tokenized]
alpha_texts_tokenized_test = [[word for word in text if word.isalpha()] for text in texts_test_tokenized]
alpha_texts_tokenized_valid = [[word for word in text if word.isalpha()] for text in texts_valid_tokenized]


#%% removing stop words

from nltk.corpus import stopwords

# here english stopwords, but it will not always work properly

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

#%% lemmatizing - slower, but better

import spacy

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

#%% vectorization - idk what it is
# TODO is that block even needed?

from sklearn.feature_extraction.text import TfidfVectorizer

# First, convert list of lists into list of strings
# data_str = [' '.join(doc) for doc in texts_lemmatized_spacy]

data_str_train = [' '.join(doc) for doc in texts_lemmatized_spacy_train]

# Create the vectorizer
vectorizer = TfidfVectorizer()

# Apply the vectorizer
X_train = vectorizer.fit_transform(data_str_train)
X_train.shape
type(X_train)

# The result is a sparse matrix representation of the documents in terms of TF-IDF features
print(vectorizer.get_feature_names_out()[:100])


# changing type of the result to data frame
X_np = X_train.todense()

X_np[X_np != 0].shape # (1, 1704970)

result = pd.DataFrame(X_np)

result = pd.DataFrame(X_np.A, columns=vectorizer.get_feature_names_out())

result.shape

#%% another tfidf (GPT generated)

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



#%% count vectorizing

from sklearn.feature_extraction.text import CountVectorizer

train_texts = [" ".join(text) for text in texts_lemmatized_spacy_train]
valid_texts = [" ".join(text) for text in texts_lemmatized_spacy_valid]
test_texts = [" ".join(text) for text in texts_lemmatized_spacy_test]

count_vectorizer = CountVectorizer()

X_train_count = count_vectorizer.fit_transform(train_texts)

X_valid_count = count_vectorizer.transform(valid_texts)
X_test_count = count_vectorizer.transform(test_texts)


#%%

from sklearn.cluster import KMeans

n_clusters = 8

kmeans = KMeans(n_clusters=n_clusters, random_state=0)

kmeans.fit(X_train)

train_preds = kmeans.predict(X_train_tfidf)
test_preds = kmeans.predict(X_test_tfidf)

cluster_centers = kmeans.cluster_centers_

#%% elbow

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X_train)
    inertia.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()



#%% PCA

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px

pca = PCA(n_components=3)
reduced_features = pca.fit_transform(X_train.toarray())

df_pca = pd.DataFrame(data = reduced_features, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

# plt.scatter(reduced_features[:,0], reduced_features[:,1], c=train_preds)
fig = px.scatter_3d(reduced_features[:,0], reduced_features[:,1], reduced_features[:,2])

reduced_cluster_centers = pca.transform(cluster_centers)

# plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
# plt.show()

fig = px.scatter_3d(reduced_features)
fig.show()

#%% PCA 3D

import plotly.express as px

# Redukcja wymiarowości do 3D
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_train.toarray())

# Konwersja do DataFrame
df_pca = pd.DataFrame(data = X_pca, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

# Dodanie informacji o klastrze do DataFrame
kmeans = KMeans(n_clusters=10) # załóżmy, że optymalna liczba klastrów wynosi 3
y_kmeans = kmeans.fit_predict(X_pca)
df_pca['Cluster'] = y_kmeans

# Tworzenie interaktywnego wykresu 3D
fig = px.scatter_3d(df_pca, x='principal component 1', y='principal component 2', z='principal component 3', color='Cluster')
fig.show()

