# -*- coding: utf-8 -*-
"""

@author: Smarak Pradhan
"""

# Python program to convert 
# JSON file to CSV 


import json 
import csv 

import gensim
import spacy
import nltk
import re

from bs4 import BeautifulSoup
import en_core_web_lg
nlp = en_core_web_lg.load()
from nltk.stem import LancasterStemmer, WordNetLemmatizer

import pandas as pd
df = pd.read_json (r'D:\VS Projects\Cell_Phones_and_Accessories_5.json',lines=True)
export_csv = df.to_csv (r'D:\VS Projects\Cell_Phones_and_Accessories_5.csv', index = None, header=True)
df_review=df['reviewText'].tolist()
corpus=df_review
len(corpus)
lemma=WordNetLemmatizer()
stop_words=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
            "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
            "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", 
            "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", 
            "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", 
            "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", 
            "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
            "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", 
            "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def preprocess(doc):
  doc=re.sub(r'\W', ' ',str(doc))
  doc = doc.lower()                 # Converting to lowercase
  cleanr = re.compile('<.*?>')
  doc = re.sub(cleanr, ' ',str(doc))        #Removing HTML tags
  doc = re.sub(r'[?|!|\'|"|#]',r'',str(doc))
  doc = re.sub(r'[.|,|)|(|\|/]',r' ',str(doc))
  doc=re.sub(r'\s+', ' ',str(doc),flags=re.I)
  doc=re.sub(r'^b\s+', ' ',str(doc))
  doc = re.sub(r'\[[0-9]*\]', ' ', doc)
  doc = re.sub(r'\s+', ' ',doc)
  # Removing special characters and digits
  doc = re.sub('[^a-zA-Z]', ' ', doc )
  doc = re.sub(r'\s+', ' ', doc)
  doc_list = nltk.sent_tokenize(doc)
  stop_words = nltk.corpus.stopwords.words('english')
  #Lemmatization
  tokens=doc.split()
  tokens=[lemma.lemmatize(word) for word in tokens]
  tokens=[word for word in tokens if word not in zip(stop_words,stop_words)]
  return tokens

"""Define a function to extract keywords"""
def get_aspects(x):
    doc=nlp(x) ## Tokenize and extract grammatical components
    preprocess(doc)
    doc=pd.Series(doc)
    doc=doc.value_counts().head().index.tolist() ## Get 5 most frequent nouns
    return doc
text="They look good and stick good! I just don't like the rounded shape because I was always bumping it and Siri kept popping up and it was irritating. I just won't buy a product like this again"

Tags=[]
for element in df_review:
    tokens=get_aspects(element)
    print(Tags)
    Tags.append(tokens)
    
Tag=Tags.to_csv(r'D:\VS Projects\seimens\Tags.csv', index = None, header=True)    
corpus=df_review
len(corpus)

processed_data=[]
for text in corpus:
    tokens=preprocess(text)
    processed_data.append(tokens)

processed_data[0:10]

from gensim import corpora

input_corpus=[input_dict.doc2bow(token,allow_update=True) for token in processed_data]
# using list comprehension 
input_corpus_done = ' '.join([str(elem) for elem in input_corpus]) 
final=input_corpus_final.value_counts().head().index.tolist() ## Get 5 most frequent nouns

