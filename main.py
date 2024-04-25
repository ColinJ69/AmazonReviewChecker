from bs4 import BeautifulSoup as bs
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import requests
import nltk
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
stopword=set(stopwords.words('english'))
import math

def get_comments(URL):


  web = requests.get(URL)
  soup = bs(web.content, 'lxml')
  data_str = ""
  lis = []
  for item in soup.find_all("div", class_="a-expander-content reviewText review-text-content a-expander-partial-collapse-content"):

    data_str = data_str + item.get_text()

    result = data_str.split("\n")
    for i in result:
      if i == '':
        result.remove(i)

  return result
  
def get_sentiment(item):
  response = requests.get("https://github.com/ColinJ69/AmazonReviewChecker/raw/main/Book%201%20(4).xlsx")
  data = pd.read_excel(response.content,usecols = [0,1])

  x = np.array(data["text"])
  y = np.array(data["tone"])

  cv = CountVectorizer()
  X = cv.fit_transform(x)
  xtrain, xtest, ytrain, ytest = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=42)
  model = BernoulliNB()
  fit = model.fit(xtrain, ytrain)

  lit = []
  e = get_reviews(item)
  for i in e:
    while len(lit) < 50:
      data = cv.transform([i]).toarray()
      output = fit.predict(data)
      lit.append(output.item())
      
  if sorted(lit)[math.ceil(len(lit) * 0.75)] == 0:
    return 0
  else:
    return 1

