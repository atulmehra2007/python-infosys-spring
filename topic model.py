
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import NMF
def letters_only(astr):
    return astr.isalpha()

cv=CountVectorizer(stop_words='english',max_features=500)

groups=fetch_20newsgroups()
cleaned=[]
all_names=set(names.words())
lemitizer=WordNetLemmatizer 
for post in groups.data:
    cleaned.append(' '.join([WordNetLemmatizer().lemmatize(word.lower()) for word in post.split()    if letters_only(word) and word not in all_names
 ]))
transformed = cv.fit_transform(cleaned)
nmf_model = NMF(n_components=100, random_state=43)
nmf_components = nmf_model.fit_transform(transformed)

for topic_idx, topic in enumerate(nmf_model.components_):
    label = '{}: '.format(topic_idx)
    print(label, " ".join([cv.get_feature_names_out()[i] for i in topic.argsort()[:-9:-1]]))