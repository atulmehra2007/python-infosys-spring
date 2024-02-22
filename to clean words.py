from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
def letters_only(astr):
    return astr.isalpha()
cv=CountVectorizer(stop_words='english', max_features=500)
groups=fetch_20newsgroups()

cleaned=[]
all_names=set(names.words())
for post in groups.data:
    cleaned.append(' '.join([WordNetLemmatizer().lemmatize(word.lower()) for word in post.split() if letters_only(word) and word not in all_names]))

tranformed=cv.fit_transform(cleaned)
print(cv.get_feature_names_out())