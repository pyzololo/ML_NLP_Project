#%% imports

import pandas as pd
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
                content = ''.join(lines[:first_blank_line_index])

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


#%% modifying time to one timezone




#%% splitting data

from sklearn.model_selection import train_test_split


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
gigatest.loc[gigatest['Language'] == np.NaN, 'Language']


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

gigatest.loc[idx[13], 'Text']

# the texts need cleaning, maybe it will remove anomalies


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
texts_tokenized = [word_tokenize(text.lower()) for text in texts ]
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
print_texsts(gigatest.loc[(df_lang['Language'] != 'other') & (df_lang['Language'] != 'en'), 'Text'], 0, 3)   

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

#%% removing stop words

from nltk.corpus import stopwords

# here englisg stopwords, but it will not always work properly
texts_tokenized_without_stopwords = [[word for word in text if word not in stopwords.words('english')] for text in alpha_texts_tokenized]


#%% lemmatizing tokens - using simple forms



#%% vectorization - adding columns for words

from sklearn.feature_extraction.text import CountVectorizer
