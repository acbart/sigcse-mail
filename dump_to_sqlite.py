# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 22:01:50 2018

@author: acbart
"""

import sqlite3
import os
import json
import codecs
from tqdm import tqdm
from datetime import timedelta, datetime
import re

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
        return None
    else:
        return actual

def strip_punctuation(s):
    return re.sub('[^0-9a-zA-Z ]+', '', s)

#%%
# Email Metadata
collected = []
for email_metadata in tqdm(os.listdir('parsed_emails/')):
    with open('parsed_emails/'+email_metadata) as email_metadata_file:
        row = json.load(email_metadata_file)
        row['id'] = email_metadata[:-5]
        collected.append(row)
emails = {c['id']: c for c in collected}
#%%
# Attachments
attachments = {}
for attachment in tqdm(os.listdir('cleaned_attachments/')):
    with codecs.open('cleaned_attachments/'+attachment, encoding='utf-8') as attachment_data:
        attachments[attachment] = attachment_data.read()
        
#%%
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

#%% Classification
import pandas as pd
types = pd.read_csv('labeled_emails.csv', 
                    header=None, names=['ID', 'Type'],
                    index_col='ID').to_dict()['Type']

#%% Produce 2 Tables

conn = sqlite3.connect('sigcse-emails.db')
c = conn.cursor()

# Thread Table
# ID, Year, Month, Week, Index
c.execute('''CREATE TABLE IF NOT EXISTS thread (id, year, month, week)''')

# Thread ID, Date, FROM, ID, Subject, Body
c.execute('''CREATE TABLE IF NOT EXISTS email 
             (id, thread_id, sent, sender, subject, body, kind)''')

# Insert contents
for d, y, m, w, i, l, a in threads_list:
    c.execute("INSERT INTO thread (id, year, month, week) "+
              "VALUES ('{id}', {year}, {month}, {week})".format(
                      id=d, year=y, month=m, week=i
                      ))
    for email in a:
        if email not in emails:
            print(email)
            continue
        metadata = emails[email]
        first_attachment = metadata['attachments'][0]
        if first_attachment not in attachments:
            print(first_attachment)
            body = ""
            kind = "Unknown"
        else:
            body = attachments[first_attachment]
            kind = types.get(first_attachment, "Unknown")
        c.execute("INSERT INTO email (id, thread_id, sent, sender, "
                   " subject, body, kind) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (email, 
                    d,
                   strptime_with_offset(metadata['date']),
                   metadata['reply'],
                   metadata['subject'],
                   body,
                   kind))

# Email Table


conn.commit()
conn.close()