import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from nltk.corpus import stopwords
from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter

plt.style.use('ggplot')
import re
from nltk.tokenize import word_tokenize
import random
import nltk
import spacy

import gensim
from gensim.summarization.textcleaner import split_sentences
import string
from tqdm import tqdm


df_train = pd.read_csv("/mnt/home/lakshmia/SummaryGen/data/raw/cnn_dailymail/train.csv", encoding='utf-8')
df_test = pd.read_csv("/mnt/home/lakshmia/SummaryGen/data/raw/cnn_dailymail/test.csv", encoding='utf-8')
df_val = pd.read_csv("/mnt/home/lakshmia/SummaryGen/data/raw/cnn_dailymail/validation.csv", encoding='utf-8')


stop_words = set(stopwords.words('english')) 
nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])


## Clean Train-set
cl_train = pd.DataFrame()
cl_test = pd.DataFrame()
cl_vali = pd.DataFrame()


from tqdm import tqdm
tqdm.pandas()

cl_train['article'] = df_train['article'].apply(lambda x: x.lower())
cl_train['highlights'] = df_train['highlights'].apply(lambda x: x.lower())

cl_train['article'] = cl_train['article'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
cl_train['highlights'] = cl_train['highlights'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

cl_train['article'] = cl_train['article'].apply(lambda x: re.sub(' +',' ',x))
cl_train['highlights'] = cl_train['highlights'].apply(lambda x: re.sub(' +',' ',x))

cl_train['article'] = cl_train['article'].progress_apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))

print("Train Completed")
cl_train.to_csv("/mnt/home/lakshmia/SummaryGen/data/processed/cl_train.csv")


cl_test['article'] = df_test['article'].apply(lambda x: x.lower())
cl_test['highlights'] = df_test['highlights'].apply(lambda x: x.lower())

cl_test['article'] = cl_test['article'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
cl_test['highlights'] = cl_test['highlights'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

cl_test['article'] = cl_test['article'].apply(lambda x: re.sub(' +',' ',x))
cl_test['highlights'] = cl_test['highlights'].apply(lambda x: re.sub(' +',' ',x))

cl_test['article'] = cl_test['article'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))

print("Test Completed")
cl_test.to_csv("/mnt/home/lakshmia/SummaryGen/data/processed/cl_test.csv")


cl_vali['article'] = df_val['article'].apply(lambda x: x.lower())
cl_vali['highlights'] = df_val['highlights'].apply(lambda x: x.lower())

cl_vali['article'] = cl_vali['article'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
cl_vali['highlights'] = cl_vali['highlights'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

cl_vali['article'] = cl_vali['article'].apply(lambda x: re.sub(' +',' ',x))
cl_vali['highlights'] = cl_vali['highlights'].apply(lambda x: re.sub(' +',' ',x))

cl_vali['article'] = cl_vali['article'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))

print("Val Completed")
cl_vali.to_csv("/mnt/home/lakshmia/SummaryGen/data/processed/cl_validation.csv")
