import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import seaborn as sn
import streamlit.components.v1 as components
import time

st.set_option('deprecation.showPyplotGlobalUse', False)

components.html(
    """
    <div style= "color: #696969; font-weight: bold; text-align: left; font-size: 40px; font-family: Trebuchet MS;" >
    <img src = "https://img.icons8.com/external-duo-tone-deni-mao/512/external-safe-healthy-and-medical-duo-tone-deni-mao.png" style="width: 40px; height: 40px"/>
    Anti Bait
    </div>
    <div style= "color: grey; text-align: left; font-size: 10px; font-family: Trebuchet MS;" >
    v.1.0.0
    </div>
    """,
    height=100,
)

with st.spinner(text="Loading model..."):
    time.sleep(5)
    st.success("Loading complete.")

df = pd.read_csv('./news.csv')

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

st.sidebar.header("About the Model")
st.sidebar.text(f'Accuracy of algorithm: {round(score*100,2)}%')

st.subheader("Detection Program")
user_input = st.text_input(
    "Enter the headline of the article you want to detect.")
data = tfidf_vectorizer.transform([user_input]).toarray()
with st.spinner(text="Detecting the validity..."):
    time.sleep(3)
    st.success("Detection complete.")
st.text("The following article is " + pac.predict(data)[0])

matrix = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
# 589 true positives, 587 true negatives, 42 false positives, and 49 false negatives
df_cm = pd.DataFrame(matrix, index=["FAKE", "REAL"], columns=[
                     "FAKE", "REAL"])
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
st.sidebar.header("Confusion Matrix")
st.sidebar.pyplot()
