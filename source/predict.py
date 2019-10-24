# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:47:46 2019

@author: USCHAS
"""

BERT_VOCAB= '../uncased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = '../uncased_L-12_H-768_A-12/bert_model.ckpt'
BERT_CONFIG = '../uncased_L-12_H-768_A-12/bert_config.json'




import os
import nltk
import pandas as pd
import bs4
import re
import bert
import datetime as dt
import pickle

from bs4 import BeautifulSoup
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

nltk.download('wordnet')
nltk.download('stopwords')


tokenization.validate_case_matches_checkpoint(True,BERT_INIT_CHKPNT)
bert_tokenizer = tokenization.FullTokenizer(
      vocab_file=BERT_VOCAB, do_lower_case=True)
print(bert_tokenizer.tokenize("This here's an example of using the BERT tokenizer"))

df =pd.DataFrame(columns=['id','text','label','split'])

NEW_MAIL_PATH= '../newmail/'

new_mail =os.listdir(NEW_MAIL_PATH)

with open(NEW_MAIL_PATH+new_mail[0],"r") as fd:
    text = fd.read()
    
df =pd.DataFrame(columns=['text'])
df.loc[0]=text


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
sno = nltk.stem.SnowballStemmer('english')
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
from tqdm import tqdm
preprocessed_synop = []

sno = nltk.stem.SnowballStemmer('english')
#stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
from tqdm import tqdm
preprocessed_synop = []
for sentance in tqdm(df['text'].values):
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    stemmed_sentence = []
    for e in sentance.split():
        if e.lower() not in stopwords:
            s=(sno.stem(lemmatizer.lemmatize(e.lower()))).encode('utf8')#lemitizing and stemming each word
            stemmed_sentence.append(s)
    sentance = b' '.join(stemmed_sentence)
    preprocessed_synop.append(sentance)
df['text']=preprocessed_synop #adding a column of CleanedText which displays the data after pre-processing of the review 
df['text']=df['text'].str.decode("utf-8")
x=df['text'] 

xTrain = pickle.load(open('../output/' + 'xTrain.pk', 'rb'))
start = datetime.now()

vectorizer = TfidfVectorizer(min_df=10,max_features=20000,smooth_idf=True,norm="l2",\
                              tokenizer=lambda x:bert_tokenizer.tokenize(x),sublinear_tf=False,ngram_range=(1,4))

xx= vectorizer.fit_transform(xTrain)
xtest = vectorizer.transform(x)
print("Time Taken to run this cell:, ", datetime.now() - start)


MODEL_PATH='../output/'
MODEL_NAME = os.listdir(MODEL_PATH)



loaded_model = pickle.load(open(MODEL_PATH+'finalized_model.sav', 'rb')) 
y_pred = loaded_model.predict(xtest)

if(y_pred==1):
    print("NOT SPAM")
else:
    print("SPAM")

        
        






