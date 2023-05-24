#%% imports

import pandas as pd
import numpy as np
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


#%% TODO extract day, month, hour to new columns

# sample_date = gigatest['NormalizedDate'].loc[10]
gigatest['NormalizedDate'] = pd.to_datetime(gigatest['NormalizedDate'])
# gigatest['Date'] = pd.to_datetime(gigatest['Date'])


# gigatest['Year'] = gigatest['NormalizedDate'].dt.year # will be probably deleted
# gigatest['Month'] = gigatest['NormalizedDate'].dt.month
gigatest['Day'] = gigatest['NormalizedDate'].dt.day
# it does not take modified hours
gigatest['Hour'] = gigatest['NormalizedDate'].dt.hour
gigatest['Minute'] = gigatest['NormalizedDate'].dt.minute

# gigatest['Year'].value_counts() # delete
# gigatest['Month'].value_counts() # delete
np.unique(gigatest['Day'].value_counts().index) # complete set
len(np.unique(gigatest['Hour'].value_counts().index)) # complete
len(np.unique(gigatest['Minute'].value_counts().index)) # complete

#%% sorting data by NormalizedDate

# df_sorted = gigatest.sort_values('NormalizedDate')
# #df_sorted['Date']

# # there is a problem here, it does not sort it properly
# for i in range(40):
#     print(df_sorted['NormalizedDate'][i])
    
# sorted(gigatest['NormalizedDate'])
# i = np.argsort(gigatest['NormalizedDate'])
# type(i)
# i = i.tolist()
# type(gigatest.loc[i, :])

df_sorted = gigatest.loc[np.argsort(gigatest['NormalizedDate']), :].reset_index()

# # I think it works properly now
# for i in range(40):
#     print(df_sorted.loc[i, 'NormalizedDate'])

#%% splitting data

from sklearn.model_selection import train_test_split

train_test_df, valid_df = train_test_split(df_sorted, test_size=0.3, shuffle=False)
train_df, test_df = train_test_split(train_test_df, test_size=0.3, shuffle=False)

# train_test_df['Date']
# valid_df['Date']

#%% detecting language of texts


from langdetect import detect

# example
lang = detect("Ein, zwei, drei, vier")
print(lang)


def detect_language(t):
    try:
        return detect(t)            
    except:
        return 'other'

# gigatest['Language'] = gigatest['Text'].apply(lambda t: detect_language(t))

train_df['Language'] = train_df['Text'].apply(lambda t: detect_language(t))
test_df['Language'] = test_df['Text'].apply(lambda t: detect_language(t))
valid_df['Language'] = valid_df['Text'].apply(lambda t: detect_language(t))

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

texts = copy.deepcopy(df_sorted['Text'])

texts_train = copy.deepcopy(train_df['Text'])
texts_test = copy.deepcopy(test_df['Text'])
texts_valid = copy.deepcopy(valid_df['Text'])

word_tokenize(texts[0]) # too many punctuation marks
word_tokenize('pw@pw.pl') # does not recognize e-mails

# finds words, numbers and emails
pattern = '([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$|\d+|\w+)'

re.findall(pattern, texts[0])
re.findall(pattern, 'pw@pw.pl') # finds emails now 

# in both cases results are similar
#texts_tokenized = [re.findall(pattern, text.lower()) for text in texts]
texts_tokenized = [word_tokenize(text.lower()) for text in texts]

texts_train_tokenized = [word_tokenize(text.lower()) for text in texts_train]
texts_test_tokenized = [word_tokenize(text.lower()) for text in texts_test]
texts_valid_tokenized = [word_tokenize(text.lower()) for text in texts_valid]
#texts_tokenized[idx[0]]

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

#text_df = pd.DataFrame(texts)['Text']
#text_df['Text'] = texts_tokenized

# summarizing punctuation marks
dots = [count_punctuation_marks(tokens, ["."]) for tokens in texts_tokenized]
commas = [count_punctuation_marks(tokens, [","]) for tokens in texts_tokenized]
qms = [count_punctuation_marks(tokens, ["?"]) for tokens in texts_tokenized]
exs = [count_punctuation_marks(tokens, ["!"]) for tokens in texts_tokenized]


dots_train = [count_punctuation_marks(tokens, ["."]) for tokens in texts_train_tokenized]
commas_train = [count_punctuation_marks(tokens, [","]) for tokens in texts_train_tokenized]
qms_train = [count_punctuation_marks(tokens, ["?"]) for tokens in texts_train_tokenized]
exs_train = [count_punctuation_marks(tokens, ["!"]) for tokens in texts_train_tokenized]

dots_test = [count_punctuation_marks(tokens, ["."]) for tokens in texts_test_tokenized]
commas_test = [count_punctuation_marks(tokens, [","]) for tokens in texts_test_tokenized]
qms_test = [count_punctuation_marks(tokens, ["?"]) for tokens in texts_test_tokenized]
exs_test = [count_punctuation_marks(tokens, ["!"]) for tokens in texts_test_tokenized]

dots_valid = [count_punctuation_marks(tokens, ["."]) for tokens in texts_valid_tokenized]
commas_valid = [count_punctuation_marks(tokens, [","]) for tokens in texts_valid_tokenized]
qms_valid = [count_punctuation_marks(tokens, ["?"]) for tokens in texts_valid_tokenized]
exs_valid = [count_punctuation_marks(tokens, ["!"]) for tokens in texts_valid_tokenized]

# summarizing other marks
characters = ["<", ">", "/", "`", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "+", "=", ";", ":", "'", "{", "}", "[", "]", "|", "\"", '"']

oths = [count_punctuation_marks(tokens, characters) for tokens in texts_tokenized]

oths_train = [count_punctuation_marks(tokens, characters) for tokens in texts_train_tokenized]
oths_test = [count_punctuation_marks(tokens, characters) for tokens in texts_test_tokenized]
oths_valid = [count_punctuation_marks(tokens, characters) for tokens in texts_valid_tokenized]

# summarizing digits
digits = [count_punctuation_marks(tokens, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]) for tokens in texts_tokenized]

digits_train = [count_punctuation_marks(tokens, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]) for tokens in texts_train_tokenized]
digits_test = [count_punctuation_marks(tokens, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]) for tokens in texts_test_tokenized]
digits_valid = [count_punctuation_marks(tokens, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]) for tokens in texts_valid_tokenized]

token_words = [len(tokens) for tokens in texts_tokenized]

token_words_train = [len(tokens) for tokens in texts_train_tokenized]
token_words_test = [len(tokens) for tokens in texts_test_tokenized]
token_words_valid = [len(tokens) for tokens in texts_valid_tokenized]

# summarizing all marks

def count_all_marks(tokens):
    count = 0
    for token in tokens:
        count += len(token)
    return count
        
all_marks = np.array([count_all_marks(tokens) for tokens in texts_tokenized])

all_marks_train = np.array([count_all_marks(tokens) for tokens in texts_train_tokenized])
all_marks_test = np.array([count_all_marks(tokens) for tokens in texts_test_tokenized])
all_marks_valid = np.array([count_all_marks(tokens) for tokens in texts_valid_tokenized])

# now we can measure the frequency of appearances, not only counts
# finally, these variables should be included in the final df

# counting frequency
#np.array([1, 2, 3]) / np.array([4, 5, 6])
#%% 

dots = np.array(dots) / all_marks

dots_train = np.array(dots_train) / all_marks_train
dots_test = np.array(dots_test) / all_marks_test
dots_valid = np.array(dots_valid) / all_marks_valid

# np.isnan(dots)
# np.argwhere(np.isnan(dots))
# dots[196]

np.nan_to_num(dots, copy=False, nan=0.0) # works inplace

np.nan_to_num(dots_train, copy=False, nan=0.0)
np.nan_to_num(dots_test, copy=False, nan=0.0)
np.nan_to_num(dots_valid, copy=False, nan=0.0)


commas = np.array(commas) / all_marks

commas_train = np.array(commas_train) / all_marks_train
commas_test = np.array(commas_test) / all_marks_test
commas_valid = np.array(commas_valid) / all_marks_valid

np.nan_to_num(commas, copy=False, nan=0.0)

np.nan_to_num(commas_train, copy=False, nan=0.0)
np.nan_to_num(commas_test, copy=False, nan=0.0)
np.nan_to_num(commas_valid, copy=False, nan=0.0)


qms = np.array(qms) / all_marks

qms_train = np.array(qms_train) / all_marks_train
qms_test = np.array(qms_test) / all_marks_test
qms_valid = np.array(qms_valid) / all_marks_valid

np.nan_to_num(qms, copy=False, nan=0.0)

np.nan_to_num(qms_train, copy=False, nan=0.0)
np.nan_to_num(qms_test, copy=False, nan=0.0)
np.nan_to_num(qms_valid, copy=False, nan=0.0)


exs = np.array(exs) / all_marks

exs_train = np.array(exs_train) / all_marks_train
exs_test = np.array(exs_test) / all_marks_test
exs_valid = np.array(exs_valid) / all_marks_valid

np.nan_to_num(exs, copy=False, nan=0.0)

np.nan_to_num(exs_train, copy=False, nan=0.0)
np.nan_to_num(exs_test, copy=False, nan=0.0)
np.nan_to_num(exs_valid, copy=False, nan=0.0)


oths = np.array(oths) / all_marks

oths_train = np.array(oths_train) / all_marks_train
oths_test = np.array(oths_test) / all_marks_test
oths_valid = np.array(oths_valid) / all_marks_valid

np.nan_to_num(oths, copy=False, nan=0.0)

np.nan_to_num(oths_train, copy=False, nan=0.0)
np.nan_to_num(oths_test, copy=False, nan=0.0)
np.nan_to_num(oths_valid, copy=False, nan=0.0)


digits = np.array(digits) / all_marks

digits_train = np.array(digits_train) / all_marks_train
digits_test = np.array(digits_test) / all_marks_test
digits_valid = np.array(digits_valid) / all_marks_valid

np.nan_to_num(digits, copy=False, nan=0.0)

np.nan_to_num(digits_train, copy=False, nan=0.0)
np.nan_to_num(digits_test, copy=False, nan=0.0)
np.nan_to_num(digits_valid, copy=False, nan=0.0)


token_words = np.array(token_words)

token_words_train = np.array(token_words_train)
token_words_test = np.array(token_words_test)
token_words_valid = np.array(token_words_valid)

words_len = all_marks / token_words

words_len_train = all_marks_train / token_words_train
words_len_test = all_marks_test / token_words_test
words_len_valid = all_marks_valid / token_words_valid

np.nan_to_num(words_len, copy=False, nan=0.0)

np.nan_to_num(words_len, copy=False, nan=0.0)
np.nan_to_num(words_len, copy=False, nan=0.0)
np.nan_to_num(words_len, copy=False, nan=0.0)

# adding features to data frame
# it should be scaled
df_additional = pd.DataFrame({"dots":dots, "commas":commas, "qms":qms, "exs":exs, "oths":oths, "digits":digits, "token_words":token_words, "words_len":words_len})

df_additional_train = pd.DataFrame({"dots":dots_train, "commas":commas_train, 
                                    "qms":qms_train, "exs":exs_train, 
                                    "oths":oths_train, "digits":digits_train, 
                                    "token_words":token_words_train, "words_len":words_len_train})
df_additional_test = pd.DataFrame({"dots":dots_test, "commas":commas_test, 
                                   "qms":qms_test, "exs":exs_test, 
                                   "oths":oths_test, "digits":digits_test, 
                                   "token_words":token_words_test, "words_len":words_len_test})
df_additional_valid = pd.DataFrame({"dots":dots_valid, "commas":commas_valid, 
                                    "qms":qms_valid, "exs":exs_valid, 
                                    "oths":oths_valid, "digits":digits_valid, 
                                    "token_words":token_words_valid, "words_len":words_len_valid})

#%% standarization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_normalized_train = scaler.fit_transform(df_additional_train)
df_normalized_test = scaler.fit_transform(df_additional_test)
df_normalized_valid = scaler.fit_transform(df_additional_valid)

# DataFrame znormalizowany
df_normalized_train = pd.DataFrame(df_normalized_train, columns=df_additional_train.columns)
df_normalized_test = pd.DataFrame(df_normalized_test, columns=df_additional_test.columns)
df_normalized_valid = pd.DataFrame(df_normalized_valid, columns=df_additional_valid.columns)


#%% getting only alphabetic marks
alpha_texts_tokenized = [[word for word in text if word.isalpha()] for text in texts_tokenized]

alpha_texts_tokenized_train = [[word for word in text if word.isalpha()] for text in texts_train_tokenized]
alpha_texts_tokenized_test = [[word for word in text if word.isalpha()] for text in texts_test_tokenized]
alpha_texts_tokenized_valid = [[word for word in text if word.isalpha()] for text in texts_valid_tokenized]

#%% skip during presentation

# example
words = ['this', 'is', 'a', 'sentence']
' '.join(words)

# testing languages again
languages = [detect_language(' '.join(text)) for text in alpha_texts_tokenized]

# TODO add languages to train, test and valid

# adding new column
df_additional['Language'] = languages

df_lang = pd.DataFrame({'Language': languages})
df_lang.value_counts()

# it was tested before sorting rows by date
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

#%% finding non english texts (also skip)

indices = df_sorted.loc[df_additional['Language'] != 'en'].index
len(indices)

df_sorted.loc[indices[25], 'Text']

def print_texsts(indices, start, stop):
    for i in range(start, stop):
            print("TEXT " + str(i))
            print(df_sorted.loc[indices[i], 'Text'])
            print("END " + str(i))
            print("INDEX: " + str(indices[i]))
            print()
    
print_texsts(indices, 110, 115)
# indices in sorted_df!
non_english_idx = {57:'fr', 882: 'arabskie jakies', 1472: 'de', 1475: 'de',
                   10840: 'jakis szwedzki', 12998: 'szwedzki', 12764: 'Sweden',
                   15704: 'de'}

print(df_sorted.loc[df_additional['Language'] == 'de', 'Text'])
print(df_sorted.loc[15704, 'Text']) # works well

np.argwhere(indices.equals(15704))

#%% translation?

#%% removing stop words

from nltk.corpus import stopwords

# here english stopwords, but it will not always work properly
# texts_tokenized_without_stopwords = [[word for word in text if word not in stopwords.words('english')] for text in alpha_texts_tokenized]

texts_tokenized_without_stopwords_train = [[word for word in text if word not in stopwords.words('english')] for text in alpha_texts_tokenized_train]
texts_tokenized_without_stopwords_test = [[word for word in text if word not in stopwords.words('english')] for text in alpha_texts_tokenized_test]
texts_tokenized_without_stopwords_valid = [[word for word in text if word not in stopwords.words('english')] for text in alpha_texts_tokenized_valid]


#%% saving list

def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename,myList)
    print("Saved successfully!")
    
# needs to be saved as np.array
# saveList(np.array(texts_tokenized_without_stopwords), "texts_tokenized_without_stopwords.npy")
saveList(np.array(texts_tokenized_without_stopwords_train), "texts_tokenized_without_stopwords_train.npy")
saveList(np.array(texts_tokenized_without_stopwords_test), "texts_tokenized_without_stopwords_test.npy")
saveList(np.array(texts_tokenized_without_stopwords_valid), "texts_tokenized_without_stopwords_valid.npy")

#%% saving 2

import pickle

def saveList2(myList, filename):
    with open(filename, 'wb') as file:
        pickle.dump(myList, file)
        
saveList2(texts_tokenized_without_stopwords_train, "texts_tokenized_without_stopwords_train.pickle")
saveList2(texts_tokenized_without_stopwords_test, "texts_tokenized_without_stopwords_test.pickle")
saveList2(texts_tokenized_without_stopwords_valid, "texts_tokenized_without_stopwords_valid.pickle")

#%% loading list

def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename, allow_pickle=True)
    return tempNumpyArray.tolist()

# loading results
texts_tokenized_without_stopwords = loadList("texts_tokenized_without_stopwords.npy")

# texts_tokenized_without_stopwords_train = loadList("texts_tokenized_without_stopwords_train.npy")
# texts_tokenized_without_stopwords_test = loadList("texts_tokenized_without_stopwords_test.npy")
# texts_tokenized_without_stopwords_valid = loadList("texts_tokenized_without_stopwords_valid.npy")

#%% loading 2

def loadList2(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    
texts_tokenized_without_stopwords_train = loadList2("texts_tokenized_without_stopwords_train.pickle")
texts_tokenized_without_stopwords_test = loadList2("texts_tokenized_without_stopwords_test.pickle")
texts_tokenized_without_stopwords_valid = loadList2("texts_tokenized_without_stopwords_valid.pickle")


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


#%% lemmatizing - slower, but better

import spacy

nlp = spacy.load('en_core_web_sm')

def lemmatize_words(doc):
    doc = nlp(" ".join(doc))
    return [token.lemma_ for token in doc]

# texts_lemmatized_spacy = list(map(lemmatize_words, texts_tokenized_without_stopwords))


texts_lemmatized_spacy_train = list(map(lemmatize_words, texts_tokenized_without_stopwords_train))
texts_lemmatized_spacy_test = list(map(lemmatize_words, texts_tokenized_without_stopwords_test))
texts_lemmatized_spacy_valid = list(map(lemmatize_words, texts_tokenized_without_stopwords_valid))

#%% vectorization

from sklearn.feature_extraction.text import TfidfVectorizer

# assuming your data is in a variable called 'data'
# where data is a list of lists with each inner list being a list of lemmatized words

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

#%% another tfidf

from sklearn.feature_extraction.text import TfidfVectorizer

# Przygotowanie danych: zmiana listy słów na pojedyncze stringi dla każdego dokumentu
train_texts = [" ".join(text) for text in texts_lemmatized_spacy_train]
valid_texts = [" ".join(text) for text in texts_lemmatized_spacy_valid]
test_texts = [" ".join(text) for text in texts_lemmatized_spacy_test]

# Inicjalizacja TfidfVectorizer
vectorizer = TfidfVectorizer()

# Trenowanie vectorizera na danych treningowych
X_train = vectorizer.fit_transform(train_texts)

# Transformacja danych walidacyjnych i testowych
X_valid = vectorizer.transform(valid_texts)
X_test = vectorizer.transform(test_texts)



#%%

from sklearn.feature_extraction.text import CountVectorizer

train_texts = [" ".join(text) for text in texts_lemmatized_spacy_train]
valid_texts = [" ".join(text) for text in texts_lemmatized_spacy_valid]
test_texts = [" ".join(text) for text in texts_lemmatized_spacy_test]

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(train_texts)

X_valid = vectorizer.transform(valid_texts)
X_test = vectorizer.transform(test_texts)


#%%

from sklearn.cluster import KMeans

n_clusters = 8

kmeans = KMeans(n_clusters=n_clusters, random_state=0)

kmeans.fit(X_train)

train_preds = kmeans.predict(X_train)
test_preds = kmeans.predict(X_test)

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

