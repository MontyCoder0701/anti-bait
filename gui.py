import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import seaborn as sn
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

st.text("Anti-Bait")

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

st.text("This algorithm is intended to discern the validity of a news article.")
st.text(f'Accuracy of algorithm: {round(score*100,2)}%')

matrix = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
# 589 true positives, 587 true negatives, 42 false positives, and 49 false negatives
df_cm = pd.DataFrame(matrix, index=["FAKE", "REAL"], columns=[
                     "FAKE", "REAL"])
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
st.pyplot()


user_input = st.text_input(
    "Enter the headline of the article you want to detect.")
data = tfidf_vectorizer.transform([user_input]).toarray()
st.text("The following article is " + pac.predict(data)[0])
