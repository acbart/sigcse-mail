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
from matplotlib.ticker import FuncFormatter
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
import string
from textstat.textstat import textstat
from stepwise import forward_selected
from states import state_names
import scipy.stats as stats
import sklearn.metrics as metrics

#%%

wordcounts = []
y_wordcounts = []
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
lengths_bigrams = pd.Series(data=[len(w) for w in ym_words_bigrams],
                    index=index)
y_lengths_bigrams = lengths_bigrams.sum(level=0)
def counts(word):
    return pd.Series(data=[w[word] for w in wordcounts],
                     index=index)
def counts_bi(phrase):
    return pd.Series(data=[p[phrase] for p in wordcounts_bigrams],
                     index=index)
#%%
BORING_PHRASES = ('computer science', 'software engineering',
                  'assistant professor', 'computer engineering',
                  'department computer', 'invites applications',
                  'professor computer', 'state university')
pd.Series(data=[[(phrase, count)
                 for phrase, count in p.most_common(100)
                 if phrase not in BORING_PHRASES]
                for p in wordcounts_bigrams],
          index=index)
#%%
def standardize_and_show_plot(ylim):
    plt.legend(loc=(1.04,.28), framealpha=0.2)
    plt.gcf().set_size_inches(4, 2)
    plt.xlim(1997, 2016)
    #plt.yticks(np.arange(0, ylim, .001))
    plt.yticks(np.arange(max(0, round(plt.gca().get_ylim()[0], 4)-.001), 
                         round(plt.gca().get_ylim()[1], 4)+.001, .001))
    plt.ylabel("Proportion")
    #plt.xlabel("Year")
    plt.show()
#%%
## Programming Languages
for keyword, language in [('java', 'Java'), 
                           ('python', 'Python'),
                           ('c', 'C'), 
                           ('cpp', 'C++')]:
    plt.plot((counts(keyword).sum(level=0))/y_lengths, 
             label=language)
standardize_and_show_plot(.005)
#%%
## Gendered words
FEMALE_TERMS = ("women", "woman", "she", "her", "herself",
              "females", "girls", "gals",
              "lady", "ladies",
               "female", "girl", "gal")
MALE_TERMS = ("men", "man", "he", "his", "him", "himself",
                 "males", "boys", "guys",
                 "gentleman", "gentlemen",
                "male", "boy", "guy")

(sum(counts(w) for w in MALE_TERMS).sum(level=0)/len(overall)).plot(label="Male")
(sum(counts(w) for w in FEMALE_TERMS).sum(level=0)/len(overall)).plot(label="Female")
plt.xlabel("")
standardize_and_show_plot(.03)
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
## PL Styles
for keyword, language in [('object-oriented', 'Object-Oriented'), 
                           ('functional', 'Functional'),
                           ('imperative', 'Imperative')]:
    plt.plot((counts(keyword).sum(level=0))/y_lengths, 
             label=language)
standardize_and_show_plot(.03)
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
for keyword, language in [('cs0', 'CS0'), 
                           ('cs1', 'CS1'),
                           ('cs2', 'CS2'),
                           ('cs3', 'CS3')]:
    plt.plot((counts(keyword).sum(level=0))/y_lengths, 
             label=language)
standardize_and_show_plot(.03)
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
for keyword, language in [('middle school', 'Middle School'), 
                           ('high school', 'High School'),
                           ('k 12', 'K-12')]:
    plt.plot((counts_bi(keyword).sum(level=0))/y_lengths_bigrams, 
             label=language)
standardize_and_show_plot(.001)
#%%

s_wordcounts = []
s_y_wordcounts = []
s_ym_words = []
s_indexes = []
s_overall = Counter()
for filename in os.listdir('monthly_subjects'):
    _, year, month = filename.split('_')
    month = int(month[:-4])
    year = int(year)
    if year in (1996, 2017):
        continue
    with codecs.open('monthly_subjects/'+filename, encoding='utf-8') as inp:
        stripped_words = [w.strip() for w in inp.readlines()]
        words = Counter(stripped_words)
        s_ym_words.append(stripped_words)
        s_wordcounts.append(words)
        s_indexes.append((year, month))
        s_overall.update(stripped_words)

s_index = pd.MultiIndex.from_tuples(s_indexes, names=('Year', "Month"))
s_lengths = pd.Series(data=[len(w) for w in s_ym_words],
                    index=s_index)
s_y_lengths = s_lengths.sum(level=0)
def s_counts(word):
    return pd.Series(data=[w[word] for w in s_wordcounts],
                     index=s_index)
#%%
BORING_PHRASES = ['faculty', 'position', 'cs', 'workshop',
                  'science', 'university', 'tenuretrack', 'sigcse']
pd.Series(data=[[(phrase, count)
                 for phrase, count in p.most_common(100)
                 if phrase not in BORING_PHRASES]
                for p in s_wordcounts],
          index=s_index).map(lambda x: ', '.join([y[0] for y in x][:10]))
