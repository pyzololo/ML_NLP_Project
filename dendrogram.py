import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import pickle
from scipy.cluster.hierarchy import linkage, dendrogram

def loadList2(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

#Z = linkage(X_train_tfidf_df, method='single')
#saveList2(Z, "Z.pickle")
Z = loadList2("Z.pickle")
dendrogram(Z)
plt.show()