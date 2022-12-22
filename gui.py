import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import seaborn as sn
import streamlit.components.v1 as components
import time
import re

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Anti-Bait", page_icon="🛡️", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)

components.html(
    """
    <div style= "color: #355E3B; font-weight: bold; text-align: left; font-size: 40px; font-family: Trebuchet MS; box-shadow: 5px 5px 5px #AFE1AF" >
    <img src = "https://img.icons8.com/fluency/512/security-shield-green.png" style="width: 40px; height: 40px"/>
    Anti Bait
    <div style= "color: grey; text-align: left; font-size: 10px; font-family: Trebuchet MS;" >
    v.1.0.0
    <br></br>
    </div>
    </div>
    """,
    height=150,
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

with st.form("my_form"):
    try:
        user_input = st.text_input(
            "Enter the headline of the article you want to detect.")
    except ValueError:
        st.error(
            "Please enter the article headline in a full english sentence format.")

    submitted = st.form_submit_button("Submit")
    if submitted:
        data = tfidf_vectorizer.transform([user_input]).toarray()
        with st.spinner(text="Detecting the validity..."):
            time.sleep(3)
            st.success("Detection complete.")
        if len(user_input) == 0:
            st.error(
                "The input is blank. Please try again.")
        else:
            match = re.search(r'[a-zA-Z]', user_input)
            if match == None:
                st.error(
                    "This is not a sentence. Please try again.")
            else:
                st.text("The following article is " + pac.predict(data)[0])
                if pac.predict(data)[0] == "UNBIASED":
                    st.balloons()

matrix = confusion_matrix(y_test, y_pred, labels=['BIASED', 'UNBIASED'])
df_cm = pd.DataFrame(matrix, index=["BIASED", "UNBIASED"], columns=[
    "BIASED", "UNBIASED"])
sn.set(font_scale=1.4)
sn.heatmap(df_cm, cmap="Greens", annot=True, annot_kws={"size": 16})
st.sidebar.header("Confusion Matrix")
st.sidebar.pyplot()
