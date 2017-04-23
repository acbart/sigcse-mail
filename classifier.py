# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 14:50:42 2017

@author: acbart
"""

import os
import codecs
import csv
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

#%%
attachments = {}
#with open('email_classification.csv', 'w', newline='') as out:
    #wr = csv.writer(out, quoting=csv.QUOTE_ALL)
files = os.listdir('cleaned_attachments/')
#random.shuffle(files)
data_names = []
data = []
for attachment in files:
    if attachment.endswith('.txt'):
        with codecs.open('cleaned_attachments/'+attachment, encoding='utf-8') as attachment_data:
            data_names.append(attachment)
            data.append(attachment_data.read())
data_names = pd.Series(data_names)
data = pd.Series(data)
paired = pd.Series(dict(zip(data_names, data)))
            #short = data[:100].replace('\n', ' ').replace('\r', ' ')
            #wr.writerow(["", attachment, short])
#%%    
training = pd.read_csv('email_classification_post.csv',
                       names=['Label', 'ID', 'Text'],
                       index_col='ID',
                       header=None)
del training['Text']
training = training.dropna(subset=['Label'])
training['text'] = paired

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
])
text_clf = text_clf.fit(training.text, training.Label)

#%%
docs_new = paired.drop(training.index)

predicted = text_clf.predict(docs_new)
predicted_paired = pd.Series(predicted, index=docs_new.index)
for doc, category in list(zip(docs_new.index, predicted))[:50]:
    print('%r => %s' % (doc, category))
predicted_paired.to_csv('labeled_emails.csv')
#%%
# Test
tested = text_clf.predict(training.text)
print(np.mean(tested == training.Label))
#%%
print(metrics.classification_report(training.Label, tested))
print(metrics.confusion_matrix(training.Label, tested))