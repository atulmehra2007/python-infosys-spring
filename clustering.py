from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
    """
    Check if the input string contains only letters.
    :param astr: The input string to be checked.
    :return: True if the input string contains only letters, False otherwise.
    """
def letters_only(astr):
    return astr.isalpha()
cv=CountVectorizer(stop_words='english', max_features=500)
groups=fetch_20newsgroups()

cleaned=[]
all_names=set(names.words())
for post in groups.data:
    cleaned.append(' '.join([WordNetLemmatizer().lemmatize(word.lower()) for word in post.split() if letters_only(word) and word not in all_names]))

tranformed=cv.fit_transform(cleaned)
km=KMeans(n_clusters=20)
km.fit(tranformed)
labels=groups.target
plt.scatter(labels,km.labels_)
plt.xlabel('news group')
plt.ylabel('cluster')
plt.show()