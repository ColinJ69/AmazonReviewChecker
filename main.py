from bs4 import BeautifulSoup as bs
import requests

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
def get_sentiment():
  response = requests.get("personal_dataset.not_public")
  data = pd.read_excel(response.content,usecols = [0,1])

  x = np.array(data["Text"])
  y = np.array(data["Label"])

  cv = CountVectorizer()
  X = cv.fit_transform(x)
  xtrain, xtest, ytrain, ytest = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=42)
  model = BernoulliNB()
  fit = model.fit(xtrain, ytrain)

  lit = []
  e = get_posts(user)
  for i in e:
    data = cv.transform([i]).toarray()
    output = fit.predict(data)
    lit.append(output.item())
  return lit

