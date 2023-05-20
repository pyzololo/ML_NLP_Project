#%% imports

import pandas as pd
import re


#%% reading data [UPDATED]

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

## NOTE
# it turns out that function that is supposed to change the time including
# time zone chuja robi, the time remains the same with, and without including 
# time zone. pierdole to. it is not that important now

#%% 8 records with no date [READ AND DELETE]

## NOTE
# this block is now irrrelevant since its all OK now
# you can read this block if you want and delete


gigatest[gigatest['NormalizedDate'].isna()]

null_date_index = gigatest[gigatest['NormalizedDate'].isna()].index

gigatest['Lines'].loc[null_date_index]

# all of these have 'Lines' value of 0

sus_records = gigatest.loc[null_date_index]

for index in null_date_index:
    print("\n============================ NEXT FILE ============================\n")
    print(gigatest['Text'].loc[index])

# it turns out, that these records are read wrongly, in all of these some 
# header info end up in the 'Text'

# the issue is that in these files some data is not in the header, but somewhere else
# reading function will be modified to search for date and lines info in whole text

## DONE

#%% TODO extract day, month, hour to new columns

# sample_date = gigatest['NormalizedDate'].loc[10]
gigatest['NormalizedDate'] = pd.to_datetime(gigatest['NormalizedDate'])
# gigatest['Date'] = pd.to_datetime(gigatest['Date'])


gigatest['Year'] = gigatest['NormalizedDate'].dt.year # will be probably deleted
gigatest['Month'] = gigatest['NormalizedDate'].dt.month
gigatest['Day'] = gigatest['NormalizedDate'].dt.day
gigatest['Hour'] = gigatest['NormalizedDate'].dt.hour
gigatest['Minute'] = gigatest['NormalizedDate'].dt.minute

#%% splitting data TODO

df_sorted = gigatest.sort_values('NormalizedDate')

#%% splitting data

from sklearn.model_selection import train_test_split

train_test_df, valid_df = train_test_split(gigatest, test_size=0.3, shuffle=False)
train_df, test_df = train_test_split(train_test_df, test_size=0.3, shuffle=False)


#%% detecting language of texts


from langdetect import detect
import numpy as np

# example
lang = detect("Ein, zwei, drei, vier")
print(lang)


def detect_language(t):
    try:
        return detect(t)            
    except:
        return 'other'

gigatest['Language'] = gigatest['Text'].apply(lambda t: detect_language(t))

# in such places there are problems, an exception is thrown
gigatest.loc[869,'Text']

#%% analyzing languages

gigatest.info()
# mostly english but not only
gigatest['Language'].value_counts()

# nothing here
gigatest.loc[gigatest['Language'] == 'other', 'Language']


idx = gigatest.loc[gigatest['Language'] == 'de', 'Language'].index

# rows with german texts
gigatest.loc[idx, 'Text']

for t in gigatest.loc[idx, 'Text']:
    print('TEXT_START')
    print(t)
    print('TEXT_END')
    print()
    

# most of these texts are strange
gigatest.loc[idx[0], 'Text']
gigatest.loc[idx[1], 'Text']

idx[0]
gigatest.loc[idx[13], 'Text']

# the texts need cleaning, maybe it will remove anomalies

# 2542     comp.os.ms-windows.misc
# 2544     comp.os.ms-windows.misc
# 2979     comp.os.ms-windows.misc
# 2980     comp.os.ms-windows.misc
# 2982     comp.os.ms-windows.misc
# 2984     comp.os.ms-windows.misc
# 2986     comp.os.ms-windows.misc
# 6777                misc.forsale
# 9738          rec.sport.baseball
# 10004           rec.sport.hockey
# 10271           rec.sport.hockey
# 11012                  sci.crypt
# 16342         talk.politics.guns

#%% word tokenization

from nltk.tokenize import word_tokenize
import copy

texts = copy.deepcopy(gigatest['Text'])

word_tokenize(texts[0]) # too many punctuation marks
word_tokenize('pw@pw.pl') # does not recognize e-mails

# finds words, numbers and emails
pattern = '([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$|\d+|\w+)'

re.findall(pattern, texts[0])
re.findall(pattern, 'pw@pw.pl') # finds emails now 

# in both cases results are similar
#texts_tokenized = [re.findall(pattern, text.lower()) for text in texts]
texts_tokenized = [word_tokenize(text.lower()) for text in texts]
texts_tokenized[idx[0]]

# before the numbers and special marks are removed, it would be nice to get some statistics from them ...


# getting only alphabetic marks
alpha_texts_tokenized = [[word for word in text if word.isalpha()] for text in texts_tokenized]


# example
words = ['this', 'is', 'a', 'sentence']
' '.join(words)

# testing languages again
languages = [detect_language(' '.join(text)) for text in alpha_texts_tokenized]

df_lang = pd.DataFrame({'Language': languages})
df_lang.value_counts()

# mostly empty texts, no words
gigatest.loc[df_lang['Language'] == 'other', 'Text']


# inspecting languages

# 77 non english texts
gigatest.loc[(df_lang['Language'] != 'other') & (df_lang['Language'] != 'en'), 'Text'].shape
        
def print_texsts(texts, start, stop):
    indexes = texts.index
    for i in range(start, stop):
            print("TEXT " + str(start))
            print(texts[indexes[i]])
            print("END " + str(start))
            print(indexes[start])
            print()
            start += 1
            
# these texts are in english too
print_texsts(gigatest.loc[(df_lang['Language'] != 'other') & (df_lang['Language'] != 'en'), 'Text'], 0, 2)   

# but this text is in german
print_texsts(gigatest.loc[df_lang['Language'] == 'de', 'Text'], 0, 1)

# there are problems with such texts
gigatest.loc[2543, 'Text']

# tokens of problematic text
texts_tokenized[2543]
len(texts_tokenized[2543])

# using only letters helps
alpha_texts_tokenized[2543]
len(alpha_texts_tokenized[2543])


# translation?

#%% removing stop words

from nltk.corpus import stopwords

# here english stopwords, but it will not always work properly
texts_tokenized_without_stopwords = [[word for word in text if word not in stopwords.words('english')] for text in alpha_texts_tokenized]


#%% saving list

def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename,myList)
    print("Saved successfully!")
    
# needs to be saved as np.array
saveList(np.array(texts_tokenized_without_stopwords), "texts_tokenized_without_stopwords.npy")
    
#%% loading list

def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename, allow_pickle=True)
    return tempNumpyArray.tolist()

# loading results
texts_tokenized_without_stopwords = loadList("texts_tokenized_without_stopwords.npy")


#%% lemmatizing tokens - using simple forms

# IN CASE OF ERROR
# import nltk
# nltk.download('wordnet')
# nltk.download('omw') ## will download newest version, did't work for me
# nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

def lemmatize_words(words):
    return [lemmatizer.lemmatize(word) for word in words]

texts_lemmatized = list(map(lemmatize_words, texts_tokenized_without_stopwords))


#%% lemmatizing - slower, but better (haven't tested it yet)

import spacy

nlp = spacy.load('en_core_web_sm')

def lemmatize_words(doc):
    doc = nlp(" ".join(doc))
    return [token.lemma_ for token in doc]

texts_lemmatized_spacy = list(map(lemmatize_words, texts_tokenized_without_stopwords))


#%% vectorization - adding columns for words

from sklearn.feature_extraction.text import CountVectorizer
