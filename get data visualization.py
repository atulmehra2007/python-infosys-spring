from sklearn.datasets import fetch_20newsgroups
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
groups=fetch_20newsgroups()
sns.displot(groups.target)
plt.show()