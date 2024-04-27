from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup as bs
import requests
import math

def get_reviews(URL):


  web = requests.get(URL)
  soup = bs(web.content, 'lxml')
  lis = []
  for item in soup.find_all("div", class_="a-expander-content reviewText review-text-content a-expander-partial-collapse-content"):


  
      lis.append(item.text)


  return lis
  
def get_sentiment(item):
    results = []
    e = get_reviews(item)
    for comment in e:
     sid = SentimentIntensityAnalyzer()
     ss = sid.polarity_scores(comment)
     for k in sorted(ss):
         print('{0}: {1}, '.format(k, ss[k]), end='')
         x = k['pos']
         y = k['neu']
         z = k['neg']
         if x > y & x > z:
          results.append(2)
         elif y > x & y > z:
          results.append(1)
         else:
          results.append(0)
    return sorted(results)[math.ceil(len(results)*0.5)]
    
