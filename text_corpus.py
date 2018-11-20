# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:10:15 2018

@author: acbart
"""

# Run process-2.py first

#%%
import pandas as pd
from gensim import corpora, models

col = metadata.body
"""Derive topic features from a text pandas series"""
# generate topics for corpora
colname = col.name
col = col.astype(str).apply(lambda x:x.split())
dictionary = corpora.Dictionary(col)
corpus = [dictionary.doc2bow(text) for text in col]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=100, passes=2, iterations=50)
lda.print_topics(-1)
# get topic distribution for doc
def get_topics(words): return dict(lda[dictionary.doc2bow(words)])
topics_df = pd.DataFrame(col.apply(get_topics).tolist()).fillna(0.001)
topics_df.columns = ['topic_'+str(cn)+'_'+colname for cn in topics_df.columns]
return topics_df