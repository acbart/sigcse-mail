import string
import re

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

def strip_punctuation(s):
    return re.sub('[^0-9a-zA-Z ]+', '', s)

stop = set(stopwords.words('english'))

exclude = list(string.punctuation)
trans_table = {ord(c): None for c in string.punctuation}
lemma = WordNetLemmatizer()

def clean_text(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def sentiment_extracter(naive_bayes=False):
    def _run_text_blob(document):
        if not naive_bayes:
            sentiment = TextBlob(document).sentiment
            return sentiment.polarity
        else:
            sentiment = TextBlob(document, analyzer=NaiveBayesAnalyzer()).sentiment
            return sentiment.p_pos
    return _run_text_blob

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
