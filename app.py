import streamlit as st
import sklearn
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from sklearn.base import BaseEstimator, TransformerMixin

class TextTransformation(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.transformation(text) for text in X]

    def transformation(self, text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        y = [i for i in text if i.isalnum()]
        y = [i for i in y if i not in self.stop_words]
        y = [self.ps.stem(i) for i in y]
        return " ".join(y)

with open('fake_news_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Fake News Classifier")

st.write("Fake news has become a prevalent issue in today's digital landscape, leading to misinformation, confusion, and distrust among the public. From false health claims to manipulated political stories, fake news can have far-reaching consequences on society. Our app addresses these challenges by providing users with reliable tools to combat fake news effectively.")


input_text=st.text_area("Enter the news here")
btn=st.button("Detect")


transformed_text = TextTransformation().transform([input_text])
result=model.predict(transformed_text)

if btn:
    if result[0]==1:
        st.success("The news is real")
    else:
        st.error("The news is fake")





