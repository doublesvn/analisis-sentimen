import streamlit as st
import emoji
import nltk
import pandas as pd
import re,string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from mpstemmer.mpstemmer import MPStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import pickle
import numpy as np
from sklearn.feature_extraction.text import  TfidfVectorizer


def cleansing(dataTweet):
    tweet_cleaning =[]
    for tweet in dataTweet:
        tweet=re.sub(r'http\S+','',tweet)
        tweet=re.sub('&amp','',tweet)
        tweet=re.sub('\n','',tweet)
        tweet=re.sub('@[^\s]+','',tweet)
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        for character in string.punctuation:
            tweet = tweet.replace(character, ' ')
        tweet=re.sub(r'\w*\d\w*', '',tweet).strip()
        tweet_cleaning.append(tweet)
    return tweet_cleaning

def caseFolding(dataTweet):
    tweet_case_folding =[]
    for tweet in dataTweet:
        tweet=tweet.lower()
        tweet_case_folding.append(tweet)
    return tweet_case_folding

def convertEmoticon(datatweet):
    tweet_convert_emoticon =[]
    for tweet in datatweet:
        tweet=emoji.demojize(tweet)
        tweet=re.sub(':',' ',tweet)
        tweet=re.sub('_','',tweet)
        tweet = re.sub('-','',tweet)
        tweet_convert_emoticon.append(tweet)
    return tweet_convert_emoticon

def stemming(dataTweet):
    tweet_stemming = []
    mpstemmer = MPStemmer()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    for tweet in dataTweet:
        tweet   = stemmer.stem(tweet)
        tweet   = mpstemmer.stem_kalimat(tweet)
        tweet_stemming.append(tweet)
    return tweet_stemming

def tokenizing(dataTweet):
    tweet_tokenize = []
    for tweet in dataTweet:
        tweet = word_tokenize(tweet)
        tweet_tokenize.append(tweet)
    return tweet_tokenize

def stopwordFiltering(dataTweet):
    new_stopwords = ["doang","sih","dah","iya","min","kek","d","yo",'lho',"si",'tl','yee','ae','hmm','yaa','wkwkwk','ajah','nya','nd','wag','kn','waduhhhh','eeh','donk','nahhhh','neh','hehe','wkwk','ehh','ieu','oalah','ahh','hadeeehhh','looolllll','ciaatt','hadehhhhh','uyyyy','kokk',"hah","ehhh","bla","niih","lha"]
    listStopword = stopwords.words('indonesian')
    listStopword.extend(new_stopwords)
    listStopword.remove('tidak')
    listStopword.remove('kurang')
    listStopword.remove('belum')

    tweet_stopword = []
    for tweet in dataTweet:
        tweet_without_stopword=[]
        for word in tweet:
            if word not in listStopword:
                tweet_without_stopword.append(word)
        tweet_stopword.append(tweet_without_stopword)
    return tweet_stopword

def convertNegation(dataTweet):
    tweet_negation = dataTweet
    negation_word_list = ["tidak",'belum','kurang']
    for tweet in tweet_negation:
        for index, kata in enumerate(tweet):
            if(kata in negation_word_list ):
                if (len(tweet) > index+1):
                    sentence = kata + "_" + tweet[index+1]
                    tweet[index] = sentence
                    del tweet[index+1]
    return tweet_negation

def preprocessing(dataTweet):
    dataTweet = cleansing(dataTweet)
    dataTweet = caseFolding(dataTweet)
    dataTweet = convertEmoticon(dataTweet)
    dataTweet = stemming(dataTweet)
    dataTweet = tokenizing(dataTweet)
    dataTweet = convertNegation(dataTweet)
    dataTweet = stopwordFiltering(dataTweet)
    return dataTweet

def myFunc(x):
  return x

tfidf_vectorizer = pickle.load(open("tfidf.pickle", "rb"))
tfidfvec = TfidfVectorizer(tokenizer=myFunc, preprocessor=myFunc, vocabulary = tfidf_vectorizer.vocabulary_)
model = pickle.load(open("model.sav", "rb"))

def classification(text):
    text_list = [text]
    hasilprepro = preprocessing(text_list)
    tfidftest = tfidfvec.fit_transform(hasilprepro)
    x = model.predict(tfidftest)
    kelas = ""
    if x[0]==0:
        kelas = "negatif"
    elif x[0] == 1:
        kelas = "positif"
    else:
        kelas = "error"
    return hasilprepro, kelas

st.write("""
# ANALISI SENTIMEN
""")
st.sidebar.header('Coba Klasifikasi')

text_input = st.sidebar.text_input("Masukkan Teks", "")

st.write("Teks Yang diinputkan : ", text_input)
if text_input != "":
    hasilprepro, kelas = classification(text_input)
    st.write("Data hasil preprocessing : ", hasilprepro)
    st.write("Prediksi Kelas : ", kelas)
