# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 15:15:52 2016

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
stop = set(stopwords.words('english') + list(string.punctuation))

timezone_map = {'cst': -600,
                'est': -500,
                'cen': -600,
                'cdt': -500,
                'edt': -400,
                'pst': -800,
                'pdt': -700,
                'cat-2': 200,
                'gmt': 0,
                'est-10': -1000,
                'est5dst': -500,
                '-dlst': 0,
                'mdt': -600,
                'mst': -700,
                '+03d0': 300,
                'utc': 0
                }

def strptime_with_offset(string, format="%a, %d %b %Y %H:%M:%S"):
    string, timezone = string.rsplit(maxsplit=1)
    timezone = timezone.replace('--', '-')
    base_dt = datetime.strptime(string, format)
    if '-' in timezone:
        timezone = '-'+timezone.rsplit('-', maxsplit=1)[1]
    elif '+' in timezone:
        timezone = "+"+timezone.rsplit('+', maxsplit=1)[1]
    if timezone.lower() in timezone_map:
        offset = timezone_map[timezone.lower()]
    else:
        offset = int(timezone)
    delta = timedelta(hours=offset/100, minutes=offset%100)
    actual = base_dt + delta
    if actual.year < 1970:
        return pd.NaT
    else:
        return pd.to_datetime(actual)

def strip_punctuation(s):
    return re.sub('[^0-9a-zA-Z ]+', '', s)

MONTH_MAP = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12
        }
NUM_MONTH = {v: k for k,v in MONTH_MAP.items()}

#%%
# Metadata
collected = []
for email_metadata in os.listdir('parsed_emails/'):
    with open('parsed_emails/'+email_metadata) as email_metadata_file:
        row = json.load(email_metadata_file)
        row['id'] = email_metadata[:-5]
        collected.append(row)
#%%
# Attachments
attachments = {}
for attachment in os.listdir('cleaned_attachments/'):
    with codecs.open('cleaned_attachments/'+attachment, encoding='utf-8') as attachment_data:
        attachments[attachment] = attachment_data.read()
#%%
# Threads
with open('threads.json') as inp:
    old_threads = json.load(inp)
threads_list = []
for k, v in old_threads.items():
    my, w, t = k.split(',')
    month, year = [e.strip() for e in my.split()]
    month = MONTH_MAP[month]
    year = int(year)
    w = int(w[5:])
    t = int(t)
    fields = (year, month, w, t)
    i = "{}/{}/{}/{}".format(*fields)
    threads_list.append((i,)+fields + (len(v), v))
threads_list.sort()
threads = pd.DataFrame(threads_list, columns=("index", "year", "month", "week", "thread_id", "quantity", "threads"))
threads.index = threads['index']
#%%
# Per thread
threads['quantity'].plot()
plt.show()
#%%
# Thread Count per Month
threads['year/month'] = (threads['year'].astype(str) + '/'+ 
                       threads['month'].map(lambda x: '{0:0>2}'.format(x)))
threads_by_ym = threads.groupby(by=['year/month'])
threads_by_ym['quantity'].sum().plot()
plt.show()
#%%
# Thread count per year
threads_by_y = threads.groupby(by=['year'])
threads_by_y['quantity'].sum().plot()
plt.show()
#%%
metadata = pd.DataFrame(collected)

metadata['from'] = metadata['from'].apply(strip_punctuation)
metadata['date'] = metadata['date'].apply(strptime_with_offset)
metadata = metadata.dropna()
metadata['from'] = metadata['from'].replace('log in to unmask', 'unlisted')

from_counts = metadata.groupby('from').count()

time_index = pd.date_range('1/1/2000', periods=9, freq='T')

metadata['year'] = metadata['date'].map(lambda x: '{0:0>2}'.format(x.year))
metadata['month'] = metadata['date'].map(lambda x: '{0:0>2}'.format(x.month))
metadata['month_name'] = metadata['date'].map(lambda x: NUM_MONTH[x.month])
metadata['hour'] = metadata['date'].map(lambda x: '{0:0>2}'.format(x.hour))
metadata['day'] = metadata['date'].map(lambda x: '{0:0>2}'.format(x.weekday()))
metadata['daily'] = metadata['date'].map(lambda x: '{0:0>2}/{1:0>2}/{2:0>2}'.format(x.year, x.month, x.day))
metadata['daily_annual'] = metadata['date'].map(lambda x: '{0:0>2}'.format(x.dayofyear))
metadata['weekly'] = metadata['date'].map(lambda x: '{0:0>2}/{1:0>2}'.format(x.year, x.week))
metadata['weeks'] = metadata['date'].map(lambda x: '{0:0>2}'.format(x.week))
metadata['year/month'] = metadata['date'].map(lambda x: '{0:0>2}/{1:0>2}'.format(x.year, x.month))
#%%
#Load in bodies
metadata['body'] = metadata['attachments'].map(lambda x: attachments.get(x[0], ''))
metadata['body'] = metadata['body'].str.strip()
metadata['length'] = metadata['body'].map(len)
#%%
# Correlation of post length and weekly quantity
weekly_count = metadata.groupby(by='weekly')['type'].count()
weekly_length = metadata.groupby(by='weekly')['length'].sum()
plt.scatter(weekly_count, weekly_length, alpha=.1)
#%%
# Weekly # of posts
weekly_all = metadata.groupby(by='weekly')['type'].count()
axs = weekly_all.plot(alpha=.5)
plt.xticks(np.arange(0, len(weekly_all)+1, 500), rotation=20)
# 12-month rolling average
pd.rolling_mean(weekly_all, window=30, center=True).plot(title='Quantity of posts over time')
plt.show()
#%%
# Week # of posts (of the year)
weeks_all = metadata.groupby(by='weeks')['type'].count()
axs = weeks_all.plot(alpha=.5)
#plt.xticks(np.arange(0, len(weeks_all)+1, 500), rotation=20)
# 12-month rolling average
#pd.rolling_mean(weekly_all, window=4, center=True).plot(title='Quantity of posts over time')
plt.show()
#%%
# Week # of posts (of the year)
daily_annual = metadata.groupby(by='daily_annual')['type'].count()
axs = daily_annual.plot(alpha=.5)
#plt.xticks(np.arange(0, len(weeks_all)+1, 500), rotation=20)
# 12-month rolling average
#pd.rolling_mean(weekly_all, window=4, center=True).plot(title='Quantity of posts over time')
plt.show()
#%%
yearly_all = metadata.groupby(by='year')['type'].count()
axs = yearly_all.plot(alpha=.5)
#plt.xticks(np.arange(0, len(yearly_all)+1, 1), rotation=20)
# 12-month rolling average
#pd.rolling_mean(yearly_all, window=30, center=True).plot(title='Quantity of posts over time')
plt.show()

#%%
# Count per year/month
year_month = metadata.groupby(by='year/month')['type'].count()
axs = year_month.plot(alpha=.5)
plt.xticks(np.arange(0, len(metadata)+1, 45), rotation=70)
# 12-month rolling average
pd.rolling_mean(year_month, window=24, center=True).plot(title='Quantity of posts over time')
plt.show()

#%%
# Plot over months
month_data = metadata.groupby(by='month')['id'].count()
month_data.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                    'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(np.arange(0, len(month_data)+1, 1))
month_data.plot(title='Aggregate # of Posts by Month of the Year')
plt.show()

#%%
# Plot over 24 hours
day_data = metadata.groupby(by='hour')['id'].count()
day_data.index = ['{}{}'.format(hh,mer)
                    for mer in ['am', 'pm']
                    for hh in [12,1,2,3,4,5,6,7,8,9,10,11]]
plt.xticks(np.arange(0, len(day_data)+1, 3))
day_data.plot(title='Aggregate # of Posts over 24 hours')
plt.show()

#%%
dow_data = metadata.groupby(by='day')['id'].count()
dow_data.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_data.plot(title='Aggregate # of Posts over the week')
plt.show()
#%%
metadata['subject'] = metadata['subject'].apply(strip_punctuation)
words = ' '.join(metadata['subject']).split()
word_counter = Counter(words)
word_data = pd.DataFrame.from_dict(word_counter, orient='index').reset_index()
#%%
languages = pd.DataFrame(columns=['Java', 'Python', 'Scheme', 'Javascript', 'Scratch', 'C'])
for g, attachments_group in metadata.groupby(by='year')['attachments']:
    blob = ''
    for attachments in attachments_group:
        for attachment in attachments:
            path = 'cleaned_attachments/'+attachment
            if os.path.exists(path):
                with open(path, 'rb') as inp:
                    blob += inp.read().decode('utf-8')
    counts= [i for i in word_tokenize(blob.lower()) if i not in stop]
    counts = TextBlob(' '.join(counts)).word_counts
    for c in languages.columns:
        languages.set_value(g, c, counts[c.lower()])
print(languages)
#%%
ps_data = pd.DataFrame(columns=['Subjectivity', 'Polarity'])
for g, attachments_group in metadata.groupby(by='year/month')['attachments']:
    polarities, subjectivities = [], []
    for attachments in attachments_group:
        for attachment in attachments:
            path = 'cleaned_attachments/'+attachment
            if os.path.exists(path):
                with open(path, 'rb') as inp:
                    blob = TextBlob(inp.read().decode('utf-8'))
                polarities.append(blob.polarity)
                subjectivities.append(blob.subjectivity)
    ps_data.set_value(g, 'Subjectivity', sum(subjectivities)/len(subjectivities))
    ps_data.set_value(g, 'Polarity', sum(polarities)/len(polarities))
print(ps_data)
ps_data.plot()
    #print(g, list(sorted(counts.items(), key = lambda e: -e[1]))[:3])
    
# Languages (java, python, racket, javascript) - compare with tiobe
# Educational Theories (behavioralism, cognitivism, constructivism, constructionist)
# Teaching, Learning, Tools
# Conferences
# Advanced topics (Compilers, AI)
# Introductory topics (loops, variables, state, pointers)
# notational machine
# Job ads
#%%
topics = pd.DataFrame(columns=['Loops', 'Variables', 'State', 'Pointers', 'Conditionals', 'Debugging'])
for g, bodies_group in metadata.groupby(by='year')['body']:
    for c in topics:
        counts_overall = 0
        blob = ""
        for body in bodies_group:
            blob += " " + str(body)
        counts= [i for i in word_tokenize(blob.lower()) if i not in stop]
        counts = TextBlob(' '.join(counts)).word_counts
        topics.set_value(g, c, counts[c.lower()])
#print(conf_data)
topics.plot()
#%%
metadata['from'].str.lower().value_counts().to_csv('emails.csv')
#%%
# Message Size
#%%
# Subject line word cloud
text = metadata['subject'].str.cat(sep=' ')
wordcloud = WordCloud(background_color="white").generate(text)
plt.figure(figsize=(40,4))
plt.imshow(wordcloud, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
#%%
# Body word cloud
text = metadata['body'].map(str).str.cat(sep=' ')
wordcloud = WordCloud(background_color="white").generate(text)
plt.figure(figsize=(40,4))
plt.imshow(wordcloud, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
#%%
#Sentiments
metadata['polarity'] = metadata['body'].apply(lambda x: None if not x else TextBlob(str(x)).polarity)
metadata['subjectivity'] = metadata['body'].apply(lambda x: None if not x else TextBlob(str(x)).subjectivity)
#%%
# Sentimental correlations
plt.scatter(metadata['polarity'], metadata['subjectivity'], alpha=.1)
plt.show()
plt.scatter(metadata['polarity'], metadata['length'], alpha=.1)
plt.show()
plt.scatter(metadata['subjectivity'], metadata['length'], alpha=.1)
plt.show()
#%%
# Conferences
conf_data = pd.DataFrame(columns=['icer', 'iticse', 'toce', 'ccsc'])
for g, bodies_group in metadata.groupby(by='year')['body']:
    for c in conf_data:
        counts = 0
        for body in bodies_group:
            counts += str(body).lower().count(c)
        conf_data.set_value(g, c, counts)
#print(conf_data)
conf_data.plot()
#%%
def is_num(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
def normalize_name(name):
    name = name.lower()
    pieces = [p for p in name.split()
              if p not in ('dr', 'doctor', 'professor', 'prof')
              and len(p) > 1
              and not is_num(p)]
    return " ".join(sorted(set(pieces)))
    
metadata['normalized_from'] = metadata['from'].map(lambda x: normalize_name(x))
#%%
long_threads = threads[ threads['quantity'] > 2 ]
unstacked_threads = long_threads.threads.apply(pd.Series).stack().reset_index(level=-1, drop=True).reset_index()
unstacked_threads = unstacked_threads.rename(columns={'index': 'thread_index'}).set_index(0)
long_metadata = metadata.merge(unstacked_threads, how='right', left_on='id', right_index=True)
#%%
long_metadata.groupby(by='reply')['polarity'].mean().dropna().sort_values()
long_metadata.groupby(by='reply')['polarity'].std().dropna().sort_values()
#%%
posters = metadata['normalized_from'].value_counts().sort_values()
top_posters = metadata[ metadata['normalized_from'].isin(posters[posters>20].index)]
metadata.groupby(by='normalized_from')['polarity'].mean().dropna().sort_values()
top_posters.merge(unstacked_threads, how='right', left_on='id', right_index=True)
#%%
# Thread length and thread count
threaded_metadata = long_metadata.groupby(by='thread_index')
plt.scatter(threaded_metadata['length'].sum(), threaded_metadata['length'].count(), alpha=.1)
#%%
text = top_posters['subject'].str.cat(sep=' ')
wordcloud = WordCloud(background_color="white").generate(text)
plt.figure(figsize=(40,4))
plt.imshow(wordcloud, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
#%%
text = top_posters['body'].map(str).str.cat(sep=' ')
wordcloud = WordCloud(background_color="white").generate(text)
plt.figure(figsize=(60,6))
plt.imshow(wordcloud, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()
#%%
thread_dates =  threaded_metadata['date']
durations = thread_dates.apply(lambda x: x.sort_values(ascending=False))
df = thread_dates.apply(lambda x: x.sort_values(ascending=False)).diff(-1)
df = df[df >= pd.Timedelta(0)]
durations = df.groupby(level=0).apply(lambda x: x.mean())
day_durations = ((pd.to_timedelta(durations,unit='d')+pd.to_timedelta(1,unit='s')) / np.timedelta64(1,'D'))
(day_durations[day_durations < 10]).plot.hist()
dropoff = ((pd.to_timedelta(df,unit='d')+pd.to_timedelta(1,unit='s')) / np.timedelta64(1,'D'))
(dropoff[dropoff < 20]).groupby(level=0).plot(alpha=.1, color='b', linestyle='None', marker='.', markersize=5)
#%%
stopwords = set(STOPWORDS)
stopwords.add("Computer")
stopwords.add("Science")

for g, bodies_group in metadata.groupby(by='month_name')['body']:
    text = ' '.join(bodies_group)
    wordcloud = WordCloud(background_color="white",
                          stopwords=stopwords).generate(text)
    #plt.figure(figsize=(40,4))
    #plt.imshow(wordcloud, cmap=plt.cm.gray, interpolation='bilinear')
    #plt.axis("off")
    #plt.savefig('wordclouds/{}.png'.format(g))
    # Uncomment to save to file
    ##wordcloud.to_file('wordclouds/{}.png'.format(g))
#%%
def ngrams(lst, n):
    tlst = lst
    while True:
        a, b = tee(tlst)
        l = tuple(islice(a, n))
        if len(l) == n:
            yield l
            next(b)
            tlst = b
        else:
            break
yearly_counts = {}
yearly_text = {}
yearly_phrases = {}
trans_table = {ord(c): None for c in string.punctuation}
for g, bodies_group in metadata.groupby(by='year')['body']:
    text = ' '.join(bodies_group).lower()
    blob = TextBlob(text)
    yearly_counts[g] = blob.word_counts
    yearly_text[g] = text
    yearly_phrases[g] = Counter(ngrams(text.translate(trans_table).split(), 2))
def word_freq(word):
    return pd.Series([blob[word]
                      for year, blob in 
                      sorted(yearly_counts.items(), key=lambda x: x[0])],
                    sorted(yearly_counts.keys()))
def phrase_freq(phrase):
    return pd.Series([blob[phrase]
                      for year, blob in 
                      sorted(yearly_phrases.items(), key=lambda x: x[0])],
                    sorted(yearly_phrases.keys()))
#%%
phrases = [(tuple(phrase.lower().split()), phrase)
            for phrase in ["Computational Thinking",
                           "Tenure Track"]]
for phrase_tuple, phrase in phrases:
    plt.plot(phrase_freq(phrase_tuple), label=phrase)
plt.legend()
#%%
from_thread_matrix = pd.DataFrame(0, columns=metadata['normalized_from'].unique(),
                                  index=metadata['normalized_from'].unique())
for g, froms in threaded_metadata['normalized_from']:
    for userA, userB in combinations(froms, 2):
        original = from_thread_matrix.get_value(userA, userB)
        from_thread_matrix.set_value(userA, userB, original+1)
L = 200
graph = nx.from_numpy_matrix(from_thread_matrix.head(L).T.head(L).values)

# Remove free floaters
outdeg = graph.degree()
to_remove = [n for n in outdeg if outdeg[n] <= 10]
graph.remove_nodes_from(to_remove)

nx.draw_networkx(graph, node_size=20, with_labels=False, alpha=.85,
                 node_color='g',
                 width=.1)
plt.axis("off")
ax = plt.gca() # to get the current axis
ax.collections[0].set_edgecolor('none') 
plt.show()