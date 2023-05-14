#%% import pandas as pd

def read_data():
    
    import os
    import re
    import pandas as pd

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
        return np.NaN

gigatest['Language'] = gigatest['Text'].apply(lambda t: detect_language(t))

i=0
for t in gigatest['Text']:
    try:
        detect(t)
    except:
        np.NaN

# in such places there are problems, an exception is thrown
gigatest.loc[869,'Text']

#%%

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


