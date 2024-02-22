from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
cv=CountVectorizer(stop_words='english', max_features=500)
groups=fetch_20newsgroups()
tranformed=cv.fit_transform(groups.data)
print(cv.get_feature_names_out())
sns.displot(np.log(tranformed.toarray().sum(axis=0)))
plt.xlabel('log count')
plt.ylabel('frequency')
plt.title('distributed plot of 500 words count')
plt.show()