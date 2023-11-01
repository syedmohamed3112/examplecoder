import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
data=pd.read_csv("/kaggle/input/one-piece-live-action-imdb-reviews/reviews.csv")
data.head()	
data.shape()
data.info()

data.isna().sum()
data['Rating']=data['Rating'].fillna(data['Rating'].mean())
print('Average rating on One piece live adaption is ',data['Rating'].mean())
value_counts = data['Rating'].value_counts()
value_counts()
data['Rating'].value_counts().plot.bar(title='Rating Graph of One Piece Live Action')
data['Review'][0]
sent_data=pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')
sent_data.head()
columns=['tweet_id', 'airline_sentiment_confidence',
       'negativereason', 'negativereason_confidence', 'airline',
       'airline_sentiment_gold', 'name', 'negativereason_gold',
       'retweet_count', 'tweet_coord', 'tweet_created',
       'tweet_location', 'user_timezone']
def clean_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [t for t in text if len(t) > 1]
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    text = ' '.join(text)
    return text
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=20000)
X = cv.fit_transform(sent_data['text']).toarray()
y=sent_data["airline_sentiment"].to_numpy()
y
print(np.unique(y))
print(np.bincount(y))


