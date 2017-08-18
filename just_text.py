# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:52:10 2017

@author: acbart
"""

import json
import os
import pandas as pd
import codecs
from unidecode import unidecode
from datetime import timedelta, datetime
from datetime import date as datetime_date
from dateutil.parser import parse
import matplotlib.pyplot as plt
import matplotlib
import re
import math
from collections import Counter
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk import word_tokenize
from itertools import tee, islice, combinations
import networkx as nx
import string
from textstat.textstat import textstat
from stepwise import forward_selected
from states import state_names
import scipy.stats as stats
import sklearn.metrics as metrics

#%%

wordcounts = []
ym_words = []
indexes = []
overall = Counter()
ym_words_bigrams = []
wordcounts_bigrams = []
overall_bigrams= Counter()
for filename in os.listdir('monthly_bodies'):
    _, year, month = filename.split('_')
    month = int(month[:-4])
    year = int(year)
    if year in (1996, 2017):
        continue
    with codecs.open('monthly_bodies/'+filename, encoding='utf-8') as inp:
        stripped_words = [w.strip() for w in inp.readlines()]
        words = Counter(stripped_words)
        ym_words.append(stripped_words)
        wordcounts.append(words)
        indexes.append((year, month))
        overall.update(stripped_words)
        # Bigrams
        stripped_bigrams = [l+' '+r for l, r in 
                            zip(stripped_words[:-1], stripped_words[1:])]
        words_bigrams = Counter(stripped_bigrams)
        ym_words_bigrams.append(stripped_bigrams)
        wordcounts_bigrams.append(words_bigrams)
        overall_bigrams.update(stripped_bigrams)

index = pd.MultiIndex.from_tuples(indexes, names=('Year', "Month"))
lengths = pd.Series(data=[len(w) for w in ym_words],
                    index=index)
y_lengths = lengths.sum(level=0)
def counts(word):
    return pd.Series(data=[w[word] for w in wordcounts],
                     index=index)
def counts_bi(phrase):
    return pd.Series(data=[p[phrase] for p in wordcounts_bigrams],
                     index=index)
#%%
for language in ['java', 'python', 'c', 'cpp', 'scratch', 'scheme']:
    plt.plot(counts(language).sum(level=0), label=language)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Language Use")
plt.show()
#%%
FEMALE_TERMS = ("women", "woman", "she", "her",
              "females", "girls", "gals",
              "lady", "ladies",
               "female", "girl", "gal")
MALE_TERMS = ("men", "man", "he", "his",
                 "males", "boys", "guys",
                 "gentleman", "gentlemen",
                "male", "boy", "guy")

sum(counts(w) for w in MALE_TERMS).sum(level=0).plot(label="Male")
sum(counts(w) for w in FEMALE_TERMS).sum(level=0).plot(label="Female")
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Gendered Language")
plt.show()
#%%
for buzzword in ["ethics", "cheating", "cheat"]:
    counts(buzzword).sum(level=0).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.show()
#%%
for buzzword in [":)", ":-)", ":(", ":-("]:
    counts(buzzword).sum(level=0).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Smiley Faces")
plt.show()
#%%
for buzzword in ["teaching", "research"]:
    counts(buzzword).sum(level=0).plot(label=buzzword)
    #(counts(buzzword).sum(level=0)/y_lengths).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Teaching vs. Research")
plt.show()
#%%
for buzzword in ["object-oriented", "functional", "imperative"]:
    counts(buzzword).sum(level=0).plot(label=buzzword)
    #(counts(buzzword).sum(level=0)/y_lengths).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Programming Languages Styles")
plt.show()
#%%
for buzzword in ["graduate", "undergraduate"]:
    counts(buzzword).sum(level=0).plot(label=buzzword)
    #(counts(buzzword).sum(level=0)/y_lengths).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Graduate vs. Undergraduate")
plt.show()
#print(stats.pearsonr(*[(counts(buzzword).sum(level=0)/y_lengths)
#        for buzzword in ["graduate", "undergraduate"]]))
#%%
## pd.DataFrame(overall.most_common(500), columns=["Word", "Frequency"]).to_csv("top_words.csv", index=False)
#%%
for buzzword in ["acm", "ieee"]:
    counts(buzzword).sum(level=0).plot(label=buzzword)
    #(counts(buzzword).sum(level=0)/y_lengths).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Professional Organizations")
plt.show()
#%%
for buzzword in ["fall", "spring", "winter", "summer"]:
    counts(buzzword).sum(level=0).plot(label=buzzword)
    #(counts(buzzword).sum(level=0)/y_lengths).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Semesters")
plt.show()
#%%
for buzzword in ["hci", "ai", "gui"]:
    counts(buzzword).sum(level=0).plot(label=buzzword)
    #(counts(buzzword).sum(level=0)/y_lengths).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Topics")
plt.show()
#%%
for phrase in ["computational thinking", "data structures", "k 12"]:
    counts_bi(phrase).sum(level=0).plot(label=phrase)
    #(counts(buzzword).sum(level=0)/y_lengths).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Subjects")
plt.show()
#%% CS Levels
for buzzword in ["cs0", "cs1", "cs2", "cs3"]:
    counts(buzzword).sum(level=0).plot(label=buzzword)
    #(counts(buzzword).sum(level=0)/y_lengths).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Topics")
plt.show()
#%%
for phrase in ["operating systems", "artificial intelligence", 
               "human-computer interaction", "computer vision",
               "game development", "data mining",
               "discrete math", "machine learning",
               "computer graphics", "parallel distributed",
               "computer architecture", "software engineering"]:
    counts_bi(phrase).sum(level=0).plot(label=phrase)
    #(counts(buzzword).sum(level=0)/y_lengths).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(6, 6)
plt.title("Subjects")
plt.show()

#%%
for phrase in ["active learning", "paired programming", "distance education",
               "peer instruction"
               ]:
    counts_bi(phrase).sum(level=0).plot(label=phrase)
    #(counts(buzzword).sum(level=0)/y_lengths).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Subjects")
plt.show()
#%%
for phrase in ["functional programming", "object-oriented programming"]:
    counts_bi(phrase).sum(level=0).plot(label=phrase)
    #(counts(buzzword).sum(level=0)/y_lengths).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Subjects")
plt.show()
#%%
for phrase in ["women computing"]:
    counts_bi(phrase).sum(level=0).plot(label=phrase)
    #(counts(buzzword).sum(level=0)/y_lengths).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Subjects")
plt.show()
#%%
for phrase in ["computer science", "software engineering"]:
    counts_bi(phrase).sum(level=0).plot(label=phrase)
    #(counts(buzzword).sum(level=0)/y_lengths).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Subjects")
plt.show()
#%%
for phrase in ["middle school", "high school", "k 12"]:
    counts_bi(phrase).sum(level=0).plot(label=phrase)
    #(counts(buzzword).sum(level=0)/y_lengths).plot(label=buzzword)
plt.legend(loc='best', framealpha=0.2)
plt.gcf().set_size_inches(4, 3)
plt.title("Age groups")
plt.show()