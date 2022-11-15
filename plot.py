import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn
from wordcloud import WordCloud
import string

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

array = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
df_cm = pd.DataFrame(array, index=["FAKE", "REAL"], columns=[
                     "FAKE", "REAL"])
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
plt.show()

fake = df[df.label == "FAKE"]
fake['title'] = fake['title'].str.lower()
all_fake = fake['title'].str.split(' ')

all_fake_cleaned = []

for title in all_fake:
    title = [x.strip(string.punctuation) for x in title]
    all_fake_cleaned.append(title)


fake_title = [" ".join(title) for title in all_fake_cleaned]
final_title = " ".join(fake_title)

wordcloud_fake = WordCloud(background_color="white",
                           colormap="gray_r").generate(final_title)

plt.figure(figsize=(20, 20))
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis("off")
plt.show()
