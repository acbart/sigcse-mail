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
from matplotlib.ticker import FuncFormatter, ScalarFormatter

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
trans_table = {ord(c): None for c in string.punctuation}
nf = 'normalized_from'
#%%
degrees = pd.read_csv('degrees.csv')
degrees['Year'] = degrees['Year'].apply(round)
degrees['Degrees'] = degrees['Degrees'].apply(lambda x: round(x/100)*100)
degrees = degrees.set_index('Year')
plt.plot(degrees['Degrees'])
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
metadata['body_id'] = metadata['attachments'].map(lambda x: x[0])
metadata['body'] = metadata['attachments'].map(lambda x: attachments.get(x[0], ''))
metadata['body'] = metadata['body'].str.strip()
metadata['length'] = metadata['body'].map(len)
#%%
types = pd.read_csv('labeled_emails.csv', 
                    header=None, names=['ID', 'Type'],
                    index_col='ID')
metadata = metadata.join(types, on='body_id')
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
yearly_all.index = yearly_all.index.map(int)
axs = yearly_all.plot(label='job posts')
plt.show()
#%%
# Job posts follow national trend in unemployment
yearly_all = metadata[metadata['Type'] == 'Job'][metadata['year'] <= '2016']
yearly_all = yearly_all.groupby(by='year')['type'].count()
yearly_all.index = yearly_all.index.map(int)
axs = yearly_all.plot(label='job posts')
# 12-month rolling average
#pd.rolling_mean(yearly_all, window=30, center=True).plot(title='Quantity of posts over time')
plt.plot(degrees['Degrees']/500, label='degrees')
start, end = axs.get_xlim()
stepsize = 1
axs.xaxis.set_ticks(np.arange(start, end, stepsize))
axs.set_xticklabels(yearly_all.index[::stepsize], rotation=45)
plt.legend()
plt.show()
#plt.plot(yearly_all/degrees['Degrees'])
#plt.show()
#%%
unrate = pd.read_csv('UNRATE.csv')
unrate['year'] = unrate['DATE'].apply(lambda x: x[:4]).astype(int)
unrate = unrate[unrate['year'] > 1995]
unrate = unrate[unrate['year'] < 2018]
unrate['month'] = unrate['DATE'].apply(lambda x: x[5:7]).astype(int)
year_ur = unrate.groupby(by='year').mean()['UNRATE']
#plt.plot(yearly_all/year_ur)
plt.scatter(year_ur, yearly_all, c=year_ur.index)
plt.show()
#%%
sy = 2009
plt.scatter(year_ur[year_ur.index<sy], yearly_all[yearly_all.index<sy],
            color='red', label='<2010')
plt.scatter(year_ur[year_ur.index>=sy], yearly_all[yearly_all.index>=sy],
            color='blue', label='>=2010')
plt.xlabel("Unemployment Rate")
plt.ylabel("# of SIGCSE Job Adds")
plt.legend()
plt.show()
#%%
yearly_all = metadata.groupby(by='year')['Type'].count()
yearly_all.index = yearly_all.index.map(int)
plt.ylabel("# of Posts")
plt.xlabel("Year")
plt.title("# of Posts per Year")
yearly_all.plot()
#%%
# Count per year/month
year_month = metadata.groupby(by='year/month')['type'].count()
axs = year_month.plot(alpha=.25, label="Actual")
# 12-month rolling average
rolling_ym = year_month.rolling(12, center=True).mean()
ax = rolling_ym.plot(title='Quantity of posts over time', 
                     label="Rolling Mean")
# 12-month expanding average
#rolling_ym = year_month.expanding(1, center=True).mean()
#ax = rolling_ym.plot(title='Quantity of posts over time', label="Expanding Mean")
m, b, r, p, e= stats.linregress(pd.Series(range(len(year_month))),
                                year_month)
xs = np.arange(0, len(year_month))
plt.plot(xs, xs*m+b, '--', linewidth=.5)

# Set up labels
plt.xticks(rotation=45) 
start, end = ax.get_xlim()
stepsize = 13*2
ax.xaxis.set_ticks(np.arange(start, end, stepsize))
ax.set_xticklabels(rolling_ym.index[::stepsize].str[:-3].map(int))
plt.legend(loc='best', framealpha=.25)
plt.gcf().set_size_inches((5, 3))
plt.ylabel("Number of Emails")
plt.xlabel("Time (Year/Months)")
#plt.title("# of Emails per Year/Month")
plt.title("")
#plt.xticks(np.arange(0, len(rolling_ym)*10+1), rotation=70)
plt.show()

#%%
# Plot over months
MONTH_NAMES =  ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                    'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_data = metadata.groupby(by='month')['id'].count()
month_data.index = MONTH_NAMES
plt.xticks(np.arange(0, len(month_data)+1, 1))
plt.xlabel("Month")
plt.ylabel("# of Posts")
month_data.plot(title='# of Posts by Month')
plt.show()
#%%
# plot over months with stdev
month_mean = metadata.groupby(by=['month', 'year'])['id'].count().mean(level=0)
month_std = metadata.groupby(by=['month', 'year'])['id'].count().std(level=0)
month_mean.index= MONTH_NAMES
month_std.index= MONTH_NAMES
month_mean.plot.bar(yerr=month_std)
plt.xlim(-1, 12)
plt.xticks(np.arange(0, len(month_data)+1, 1), MONTH_NAMES)
plt.show()
#%%
metadata.groupby(by=['month', 'year'])['id'].count().reset_index().boxplot(column='id', by='month')
plt.title("")#"Distribution of Emails per Month")
plt.xticks(rotation=45) 
plt.xlabel("Month")
plt.ylabel("# of Emails")
plt.suptitle("")
plt.xticks(np.arange(1, len(month_data)+1, 1), MONTH_NAMES)
plt.gcf().set_size_inches((5, 3))
plt.show()
#%%
k = metadata.groupby(by=['month', 'year'])['id'].count().reset_index()
plt.xlim(0, 12)
plt.xticks(np.arange(1, 13, 1), 
           MONTH_NAMES)
plt.hist2d(list(map(int, k['month'])), 
            list(map(int, k['id'])), 
            norm=matplotlib.colors.LogNorm(), 
            bins=[12, 10])
#%%
# Plot over 24 hours
all_hours = ['00','01','02','03','04','05','06','07','08','09','10','11',
             '12','13','14','15','16','17','18','19','20','21','22','23']
all_days = pd.date_range('1996/10/02', '2018/09/10')
all_days = all_days.map(lambda x: '{0:0>2}/{1:0>2}/{2:0>2}'.format(x.year, x.month, x.day))
times_index = pd.MultiIndex.from_product([all_hours, all_days])
# GROPU BY REINDEX
day_counts = metadata.groupby(by=['hour', 'daily'])['id'].count()
day_counts = day_counts.reindex(times_index, fill_value=0)
day_data = day_counts.mean(level=0)
time_labels = ['{}{}'.format(hh,mer)
                    for mer in ['am', 'pm']
                    for hh in [12,1,2,3,4,5,6,7,8,9,10,11]]
day_data.index = time_labels
day_stds = day_counts.mad(level=0)
day_stds.index = day_data.index
plt.xticks(np.arange(0, len(day_data)+1, 3))
day_data.plot(title='Mean # Posts over 24 hours', yerr=day_stds)
plt.ylabel("# of Posts")
plt.xlabel("Time of Day")
plt.show()

k = (day_counts).reset_index()
plt.xticks(np.arange(1, 1+len(day_data)+1, 3), 
           time_labels[1::3])
plt.xlabel("Time of Day")
plt.ylabel("Email Frequency")
#plt.title("Frequency Distribution of Emails Sent during Hours of the Day")
plt.hist2d(list(map(int, k['level_0'])), 
            list(map(int, k['id'])), 
            norm=matplotlib.colors.LogNorm(), 
            bins=[24,8])
plt.gcf().set_size_inches((5, 2))
plt.colorbar()
plt.show()

#%%
all_wnumerics = ['00', '01', '02', '03', '04', '05', '06']
all_weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
daily_index = pd.MultiIndex.from_product([all_wnumerics, all_days])
dow_data = metadata.groupby(by=['day', 'daily'])['id'].count()
dow_data = dow_data.reindex(daily_index, fill_value=0)
#dow_mean.index = all_weekdays
k = dow_data.reset_index()
plt.xticks(np.arange(0, len(all_weekdays)+1, 1), 
           all_weekdays)
plt.xlabel("Day of Week")
plt.ylabel("Email Frequency")
#plt.title("Frequency Distribution of Emails Sent during Days of the Week")
plt.gcf().set_size_inches((5, 2))
plt.hist2d(list(map(int, k['level_0'])), 
            list(map(int, k['id'])), 
            norm=matplotlib.colors.LogNorm(),
            bins=[len(all_weekdays), 20])
plt.colorbar()
plt.show()
#%%
metadata['subject'] = metadata['subject'].apply(strip_punctuation)
words = ' '.join(metadata['subject']).split()
word_counter = Counter(words)
word_data = pd.DataFrame.from_dict(word_counter, orient='index').reset_index()
#%%
'''
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
'''
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
'''
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
'''
#%%
metadata['from'].str.lower().value_counts().to_csv('emails.csv')
#%%
# Message Size
#%%
# Subject line word cloud
metadata_n = metadata[metadata['Type'] == 'Normal']
text = " ".join(metadata_n['subject'].str.cat(sep=' ').lower().split())
wordcloud = WordCloud(background_color="white", 
                      prefer_horizontal=1,
                      width=500, height=500,
                      collocations =False).generate(text)
wordcloud.to_file('wordclouds/subjects.png')
#%%
# Body word cloud
metadata_n = metadata[metadata['Type'] == 'Normal']
text = metadata_n['body'].map(str).str.cat(sep=' ').lower().replace("c++", "cpp").replace("c#", "csharp")
text = text.translate(trans_table)
text = ' '.join([x for x in text.split() if len(x) > 1])
wc = WordCloud(background_color="white", 
                      prefer_horizontal=1,
                      width=500, height=500,
                      collocations =False)
wordcloud = wc.generate(text)
wordcloud.to_file('wordclouds/bodies.png')
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
plt.show()
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
for g, bodies_group in metadata.groupby(by='year')['body']:
    text = ' '.join(bodies_group).lower().replace('c++', 'cpp').replace('c#', 'csharp')
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
yearly_words = pd.Series(yearly_counts).apply(lambda x: sum(x.values()))
#%%
phrases = [(tuple(phrase.lower().split()), phrase)
            for phrase in ["Computational Thinking",
                           "Tenure Track"]]
for phrase_tuple, phrase in phrases:
    plt.plot(phrase_freq(phrase_tuple), label=phrase)
plt.legend()
plt.show()
#%%
all_languages = pd.DataFrame(columns=[])
for language in ['java', 'python', 'cpp', 'c',
                 'scratch']:
    all_languages[language] = wf = word_freq(language)/yearly_words
    wf[1:-1].plot(label=language)
for languages in [('scheme', 'racket')]:
    combined = word_freq(languages[0])
    for l_other in languages[1:]:
        combined += word_freq(l_other)
    combined = combined / yearly_words
    label = '/'.join(languages)
    combined[1:-1].plot(label=label)
    all_languages[label] = combined
plt.legend(loc=(1.1,0))
plt.title("References to Languages over Time")
plt.ylabel("Word Usage Rate")
plt.xlabel("Year")
plt.show()
#all_languages.rank(axis=1).plot()
#plt.legend(loc=(1.1,0))
#plt.show()
#%%
# Famous people
for word in ('turing','lovelace', 'hopper'):
    (word_freq(word)/yearly_words).plot(label=word)
plt.legend()
plt.show()

#%%
# Key Words
for word in ('k-12', 'diversity', 'scale', 
             'engagement'):
    (word_freq(word)/yearly_words).plot(label=word)
plt.legend()
plt.show()
#%%
# Conferences
for word in ['icer', 'iticse', 'toce', 'ccsc', 'koli']:
    (word_freq(word)/yearly_words).plot(label=word)
plt.legend(loc=(1.1,0))
plt.show()
#%%
combined = pd.DataFrame(columns=[])
for state in state_names:
    if ' ' not in state:
        combined[state] = (word_freq(state)/yearly_words)
combined.plot()
plt.legend()
plt.show()
#%%
# Gendered words
for key, words in [("female term", ("women", "woman", "she", "her",
                              "females", "girls", "gals",
                              "lady", "ladies",
                               "female", "girl", "gal")),
                   ("male term", ("men", "man", "he", "his",
                             "males", "boys", "guys",
                             "gentleman", "gentlemen",
                            "male", "boy", "guy"))]:
    combined = word_freq(words[0])
    for l_other in words[1:]:
        combined += word_freq(l_other)
    combined = combined  / yearly_words
    combined[:-1].plot(label=key)
plt.legend(loc=(1.1,0))
plt.title("Use of Gendered Language in Emails")
plt.ylabel("Word Usage Rate")
plt.xlabel("Year")
plt.show()
#%%
# Key Terms
for key, words in [("ethics", ("ethical", "ethics", "ethic")),
                   ("cheating", ("cheat", "cheating", "cheat",
                                 "plagarism")),
                   ("exam", ("exam", "examination")),
                   ("accreditation", ("accreditation",))]:
    combined = word_freq(words[0])
    for l_other in words[1:]:
        combined += word_freq(l_other)
    combined = combined / yearly_words
    combined[1:-1].plot(label=key)
plt.legend(loc=(1.1,0))
plt.title("Use of Gendered Language in Emails")
plt.ylabel("Word Usage Rate")
plt.xlabel("Year")
plt.show()

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
#%%  Yearly Jobs and Conferences
type_ym = metadata.groupby(by=['year', 'month'])['Type'].value_counts()
type_year = metadata.groupby(by='year')['Type'].value_counts().unstack()
plt.plot(type_year['Job'], label='Job')
plt.plot(type_year['Conference'], label='Conference')
plt.legend(loc='best', framealpha=0.2, ncol=2)
#plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
plt.gcf().set_size_inches(5, 2)
plt.xlabel("Year")
plt.ylabel("Posts")
#type_year.plot()
plt.xlim(1997, 2016)
plt.show()
#%%
type_month = metadata.groupby(by='month')['Type'].value_counts().unstack()
ax = type_month.plot()
ax.set_xticks(np.arange(len(NUM_MONTH)))
ax.set_xticklabels(NUM_MONTH.values(), rotation=45)
plt.show()
#%%
# Jobs as a box plot over months
ax = type_ym.unstack()['Job'].unstack().plot.box()
ax.set_xticks(np.arange(1, 1+len(NUM_MONTH)))
ax.set_xticklabels(NUM_MONTH.values(), rotation=45)
plt.gcf().set_size_inches((5, 2))
#plt.title("Distribution of Job Emails per Month")
plt.ylabel("Job Ads")
plt.show()
#%%
text = ' '.join(metadata[metadata['Type'] == 'Normal']['body'])
wordcloud = WordCloud(background_color="white",
                      stopwords=stopwords).generate(text)
plt.figure(figsize=(40,4))
plt.imshow(wordcloud, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
#%%
# Polarity has stayed constant
md_normal = metadata[metadata['Type'] == 'Normal']
ax = md_normal.groupby(by='year')['polarity'].mean().plot()
ax.set_ylim(-1, 1)
plt.show()

plt.scatter(md_normal.groupby(by='daily')['length'].mean(),
            md_normal.groupby(by='daily')['length'].count())
#ax.set_ylim(-1, 1)
plt.show()
#%%
# number of questions
metadata['questions'] = metadata[metadata['Type'] == 'Normal']['body'].str.count('\?')
metadata.groupby(by='year')['questions'].mean().plot(label='questions')
# number of exclaims
metadata['exclaims'] = metadata[metadata['Type'] == 'Normal']['body'].str.count('\!')
metadata.groupby(by='year')['exclaims'].mean().plot(label='exclaims')
plt.legend()
#%%
thread_firsts = threads.copy()
thread_firsts['op'] = thread_firsts['threads'].apply(lambda x: x[0])
threads_firsts = thread_firsts[['op', 'quantity']]
threads_firsts.set_index('op', inplace=True)
first_posts = metadata.join(threads_firsts, on='id', how='inner')
#%%
CLASSIFY_DAY = {'00': 'Day', '01': 'Day', '02': 'Day',
                '03': 'Day', '04': 'Day',
                '05': 'End', '06': 'End'}
CLASSIFY_HOUR = {'01': 'SmallHours', '02': 'SmallHours',
                 '03': 'SmallHours', '04': 'SmallHours',
                 '05': 'Morning', '06': 'Morning',
                 '07': 'Morning', '08': 'Morning',
                 '09': 'Morning', '10': 'Morning',
                 '11': 'Evening', '12': 'Evening',
                 '13': 'Evening', '14': 'Evening',
                 '15': 'Evening', '16': 'Evening',
                 '17': 'Evening',
                 '18': 'Night', '19': 'Night',
                 '20': 'Night', '21': 'Night',
                 '22': 'Night', '23': 'Night'}
CLASSIFY_MONTH = {'01': 'WINTER',
                  '02': 'WINTER',
                  '03': 'WINTER',
                  '04': 'SPRING',
                  '05': 'SPRING',
                  '06': 'SUMMER',
                  '07': 'SUMMER',
                  '08': 'SUMMER',
                  '09': 'FALL',
                  '10': 'FALL',
                  '11': 'FALL',
                  '12': 'FALL'}
def CLASSIFY_QUANTITY(quantity):
    if quantity == 1:
        return 'SINGLE'
    elif quantity <= 5:
        return 'SHORT'
    else:
        return 'LONG'
variables = ['quantity','questions','exclaims', 
             'hour', 'month', 'day',
             'length']

data = first_posts[first_posts['Type']=='Normal'][variables]
data_qual = data.copy()
data_qual['hour'] = data_qual['hour'].map(CLASSIFY_HOUR)
data_qual['day'] = data_qual['day'].map(CLASSIFY_DAY)
data_qual['month'] = data_qual['month'].map(CLASSIFY_MONTH)
data_qual['quantity'] = data_qual['quantity'].apply(CLASSIFY_QUANTITY)

data.to_csv('predicting_posts.csv', index=False)
data_qual.to_csv('predicting_posts_qual.csv', index=False)
model = forward_selected(data, 'quantity')
print(model.model.formula)
print(model.rsquared_adj)
#%%
# Distribution of posts per poster
posts_per_poster = metadata.groupby(by=nf)[nf].count()
#posts_per_poster= posts_per_poster[posts_per_poster>= 0].apply(lambda x: np.log10(x) if x > 0 else 0)
max_value = posts_per_poster.max()
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.hist(posts_per_poster, bins=np.arange(1, 150, 15), log=True, 
         histtype='bar', ec='black')
xfmt = matplotlib.ticker.LogFormatterMathtext()
plt.gcf().set_size_inches((5, 2))
plt.xlim((0, 120))
#xfmt.set_powerlimits((-3,3))
#xfmt.set_useOffset(100)
#plt.xlim(0, 5)
#plt.xticks(np.arange(0, 3, .25), 
#           ["10$^{}$".format(x) for x in np.arange(0, 3, .25)])
#posts_per_poster.plot.hist(log=True, bins=range(1, 200, 10))
plt.xlabel('Number of Posts')
plt.ylabel("Number of Posters")
plt.show()
#%% Distribution of Thread Sizes
threads.quantity.plot.hist(log=True, bins=20, histtype='bar', ec='black')
plt.xlim(0, 45)
plt.gcf().set_size_inches(5, 2)
plt.ylabel("Number of Threads")
plt.xlabel("Number of Posts")
plt.show()
#%%
#(metadata['length']).plot.hist(log=True)
#plt.gca().set_xscale("log", nonposx='clip')
data = metadata['length'].dropna()
data = data[data >= 0].apply(lambda x: np.log10(x) if x > 0 else 0)
max_value = data.max()
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.hist(data, log=True,histtype='bar', ec='black')
xfmt = matplotlib.ticker.LogFormatterMathtext()
#xfmt.set_powerlimits((-3,3))
#xfmt.set_useOffset(100)
#plt.xlim(0, 5)
plt.xticks(range(0, 6), ["10$^{}$".format(x) for x in range(0, 6)])
#plt.gca().xaxis.set_major_formatter(xfmt)
#plt.gca().set_xscale('log')
plt.gcf().set_size_inches(5, 2)
plt.ylabel("Number of Posts")
plt.xlabel("Number of Characters")
plt.show()
#%%
# of posts
print("# of posts:", len(metadata))
# of unique posters
print("# of users:", len(metadata[nf].unique()))
# Quiets
quiets = metadata[metadata[nf].isin(posters[(posters<=2)].index)]
print("# of quiets", len(quiets[nf].unique()))
print("# of quiets posts", len(quiets[nf]))
print("quiets as % of users", 100*len(quiets[nf].unique())/len(metadata[nf].unique()))
print("quiets' % of posts:", 100*len(quiets[nf])/len(metadata))
# Superactive
#   % of users
#   % of posts
# Regular
regulars = metadata[metadata[nf].isin(posters[(posters>2) & (posters<=20)].index)]
print("# of regulars", len(regulars[nf].unique()))
print("# of regulars posts", len(regulars[nf]))
print("regulars as % of users", 100*len(regulars[nf].unique())/len(metadata[nf].unique()))
print("regulars' % of posts:", 100*len(regulars[nf])/len(metadata))
# Active
#   % of users
#   % of posts
actives = metadata[metadata[nf].isin(posters[(posters>20) & (posters<=50)].index)]
print("# of actives", len(actives[nf].unique()))
print("# of actives posts", len(actives[nf]))
print("Actives as % of users", 100*len(actives[nf].unique())/len(metadata[nf].unique()))
print("Actives' % of posts:", 100*len(actives[nf])/len(metadata))
# Super
superactives = metadata[metadata[nf].isin(posters[posters>50].index)]
print("# of supers", len(superactives[nf].unique()))
print("# of supers posts", len(superactives[nf]))
print("Supers as % of users", 100*len(superactives[nf].unique())/len(metadata[nf].unique()))
print("Supers' % of posts:", 100*len(superactives[nf])/len(metadata))
#%%
thread_dates =  threaded_metadata['date']
durations = thread_dates.apply(lambda x: x.sort_values(ascending=False))
df = thread_dates.apply(lambda x: x.sort_values(ascending=False)).diff(-1)
df = df[df.groupby(level=0).cumcount(ascending=False) > 0]
df_d = ((pd.to_timedelta(df,unit='d')+pd.to_timedelta(1,unit='s')) / np.timedelta64(1,'D'))
#%%
lengths = df.groupby(level=0).apply(lambda x: x.sum())
day_lengths = ((pd.to_timedelta(lengths,unit='d')+pd.to_timedelta(1,unit='s')) / np.timedelta64(1,'D'))
day_lengths = day_lengths[day_lengths<10]
day_lengths.plot.hist(log=False, label='Durations')
h = np.histogram(day_lengths)
x = np.linspace(0, len(h[0]), len(h[0]))
from scipy.optimize import curve_fit
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
popt, pcov = curve_fit(func, x, h[0])
y = func(x, *popt)

plt.plot(x, y, 'r-', label="$y=\\frac{1}{\\sqrt{e^{x}}}$")
plt.xlabel("Days")
plt.ylabel("# of threads")
plt.title("Duration of Non-trivial Threads")
plt.ylim(0, 400)
plt.legend()
plt.show()
def goodness_of_fit(y, y_predicted):
    # residual sum of squares
    ss_res = np.sum((y - y_predicted) ** 2)
    # total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    # r-squared
    return 1 - (ss_res / ss_tot)
r2 = goodness_of_fit(h[0], y)
print("R^2 is", r2)
print("Decay factor is", -popt[1])
#%%
# Let's make a horizontal chart that shows frequency
progress = df_d[df_d< 30].groupby(level=0).cumsum()
progress = progress[progress<10]
plt.figure(figsize=(10, 1))
plt.yticks([])
plt.xticks(range(0, 1+int(progress.max()), 1))
for x in progress:
    plt.axvline(x=x, color='blue', alpha=.05)
plt.show()
#%% Duration of Non-trivial Threads
BINS = (1+int(progress.max()))*24
axs = progress.hist(bins=BINS, cumulative=-1, normed=True,
                    histtype='bar', ec='black', lw=.5)
heights = [p._height for p in axs.patches]
plt.xticks(range(0, 10, 1))
plt.xlim(0, 7)
plt.ylim(0,1)
plt.xlabel("Thread Duration (Days)")
plt.ylabel("Percentage of Threads")
xs = np.arange(0,10,1/24)
ys = [1/(2.1**x) for x in xs]
#per_formatter = FuncFormatter(lambda x: x+"%")
#plt.gca().yaxis.set_major_formatter(per_formatter)
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
plt.gcf().set_size_inches((5, 2))
plt.plot(xs, ys, color='r', label="Predicted")
plt.show()
print("MSE:", metrics.mean_squared_error(pd.Series(heights), pd.Series(ys)))
#%%
maxes = df.groupby(level=0).apply(lambda x: x.max())
day_maxes= ((pd.to_timedelta(maxes,unit='d')+pd.to_timedelta(1,unit='s')) / np.timedelta64(1,'D'))
day_maxes = day_maxes[day_maxes<10]
day_maxes.plot.hist(log=False)
#%%
durations = df.groupby(level=0).apply(lambda x: x.mean())
day_durations = ((pd.to_timedelta(durations,unit='d')+pd.to_timedelta(1,unit='s')) / np.timedelta64(1,'D'))
day_durations = day_durations[day_durations<10]
(day_durations).plot.hist(log=True)
plt.show()
durations_std = df.groupby(level=0).apply(lambda x: x.std())
day_durations_std = ((pd.to_timedelta(durations_std,unit='d')+pd.to_timedelta(1,unit='s')) / np.timedelta64(1,'D'))
day_durations_std = day_durations_std[day_durations_std<10]
day_durations_std.plot.hist(log=True)
plt.show()
plt.scatter(day_durations, day_durations_std)
plt.xlabel("Mean Durations")
plt.ylabel("StdDev Durations")
plt.show()
#%%
dropoff = ((pd.to_timedelta(df,unit='d')+pd.to_timedelta(1,unit='s')) / np.timedelta64(1,'D'))
(dropoff[dropoff < 20]).groupby(level=0).plot(alpha=.1, color='b', linestyle='None', marker='.', markersize=5)
plt.show()
threads.join((df_d.sum(level=0)[df_d.sum(level=0) > 100]), how='right')
#%%
activity_productivity = pd.read_csv("top_posters.csv")
activity_productivity = activity_productivity[activity_productivity['Papers'] > 0]
active = activity_productivity[activity_productivity['Posts'] <= 50]
supers = activity_productivity[activity_productivity['Posts'] > 50]
plt.scatter(active['Posts'], active['Papers'],
            alpha=.5, color='b', label='Active')
plt.scatter(supers['Posts'], supers['Papers'],
            alpha=.5, color='r', label='Super')
plt.ylim(0, 120)
plt.xlim(0, 350)
plt.legend(loc='best', framealpha=.25)
plt.gcf().set_size_inches((5, 3))
plt.xlabel("Number of Posts")
plt.ylabel("Number of Papers")
#plt.title("Productivity vs. Activity of Active Users")
plt.show()
#%%
# classify thread length
# 1, 2-3, 4-10, 11+
# Single, Short, Medium, Long
def length_classifier(q):
    if q <= 1:
        return 'single'
    elif 2 <= q <= 4:
        return 'short'
    else:
        return 'long'
df_firsts = pd.DataFrame(columns=[])
df_firsts['length'] = first_posts['quantity'].map(length_classifier)
df_firsts['subject'] = first_posts['subject']
df_firsts['body'] = first_posts['body']
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))
vectors = vectorizer.fit_transform(df_firsts['subject'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vectors, df_firsts['length'], 
                                                    test_size=0.33, random_state=42)
# initialise the SVM classifier
from sklearn.svm import LinearSVC
classifier = LinearSVC()
# train the classifier
classifier.fit(X_train, y_train)
# Predict
preds = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, preds))
from sklearn.metrics import classification_report
print(classification_report(y_test, preds))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, preds))
#%%
metadata.groupby(by=['month', 'year'])['id'].count().reset_index().boxplot(column='id', by='month')
plt.title("Distribution of Job Emails per Month")
plt.xlabel("Month")
plt.ylabel("# of Emails")
plt.suptitle("")
plt.xticks(np.arange(1, len(month_data)+1, 1), MONTH_NAMES)
plt.show()
#%%
from rake_nltk import Rake
r = Rake()
for year, text in yearly_text.items():
    r.extract_keywords_from_text(text)
    print(year)
    print(r.get_ranked_phrases())
    
#%%
# Key Words
for word in ('ethic', 'ethics', 
             'ethical'):
    (word_freq(word)/yearly_words).plot(label=word)
plt.legend()
plt.show()
#%%
MALE_TERMS = ("women", "woman", "she", "her", "herself",
              "females", "girls", "gals",
              "lady", "ladies",
               "female", "girl", "gal")
FEMALE_TERMS = ("men", "man", "he", "his", "him",
                "himself",
                 "males", "boys", "guys",
                 "gentleman", "gentlemen",
                "male", "boy", "guy")
if True:
    gendered_stop = [s for s in stop
                     if s.lower() not in MALE_TERMS + FEMALE_TERMS]
    from nltk.tokenize import TweetTokenizer
    tknzr = TweetTokenizer()
    for g, bodies_group in metadata.groupby(by=['year', 'month'])['body']:
        filename= "monthly_bodies/body_{}_{}.txt".format(*g)
        with codecs.open(filename, 'wb', encoding='utf-8') as out:
            for body in bodies_group:
                body = body.replace("c++", "cpp").replace("C++", "cpp")
                words = tknzr.tokenize(body)
                for word in words:
                    if word.lower() not in gendered_stop:
                        out.write(word.lower()+"\n")
    for g, bodies_group in metadata.groupby(by=['year', 'month'])['subject']:
        filename= "monthly_subjects/subject_{}_{}.txt".format(*g)
        with codecs.open(filename, 'wb', encoding='utf-8') as out:
            for body in bodies_group:
                body = body.replace("c++", "cpp")
                words = tknzr.tokenize(body)
                for word in words:
                    if word.lower() not in gendered_stop:
                        out.write(word.lower()+"\n")
#%%
biggest_threads = long_metadata.groupby(by=['year/month', 'thread_index']).size().groupby(level=0).idxmax().values
biggest_threads = [x[1] for x in biggest_threads.tolist()]
len(biggest_threads)
#%%
def remove_prefix(text, prefix):
    if text.lower().startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever
big_threads= threads[threads.quantity > 20].threads
big_posts = big_threads.apply(lambda x: x[0]).values
thread_quantities = dict(zip(big_threads.apply(lambda x: x[0]).values,
                             big_threads.apply(len).values))
big_subjects = metadata[metadata.id.isin(big_posts)].sort_values(by=['date'])
big_subjects['quantity'] = big_subjects.id.replace(thread_quantities)
top_subjects = big_subjects[['year', 'quantity', 'subject']]
top_subjects['subject'] = top_subjects['subject'].apply(lambda bs: remove_prefix(bs, "re "))
with open('top_subjects.tex', 'w') as out:
    with pd.option_context('display.max_colwidth', -1):
        top_subjects.to_latex(buf=out, index=False, column_format='lp{3cm}')