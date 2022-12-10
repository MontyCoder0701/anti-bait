import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import unittest
import streamlit as st

df = pd.read_csv('./news.csv')

df.shape
df.head()

labels = df.label
labels.head()

x_train, x_test, y_train, y_test = train_test_split(
    df['text'], labels, test_size=0.2, random_state=7)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print("This algorithm is intended to discern the validity of a news article.")
print(f'Accuracy of algorithm: {round(score*100,2)}%')

matrix = confusion_matrix(y_test, y_pred, labels=['BIASED', 'UNBIASED'])
# 589 true positives, 587 true negatives, 42 false positives, and 49 false negatives

print("Enter the headline of the article you want to detect.")
user_input = str(input())
data = tfidf_vectorizer.transform([user_input]).toarray()
print("The following article is " + pac.predict(data)[0])


class TestNews(unittest.TestCase):
    def test_real(self):
        real_news = "Ukraine: Austrian leader, Putin meet…other new developments"
        data = tfidf_vectorizer.transform([real_news]).toarray()
        self.assertEqual(pac.predict(data)[0], 'UNBIASED')

    def test_fake(self):
        fake_news = "Putin summons the soul of Hitler"
        data = tfidf_vectorizer.transform([fake_news]).toarray()
        self.assertEqual(pac.predict(data)[0], 'BIASED')
