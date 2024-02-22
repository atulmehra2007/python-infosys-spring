from sklearn.datasets import fetch_20newsgroups
import numpy as np
groups=fetch_20newsgroups()
print(groups)
print(groups.keys())
print('\n')
print(groups['target_names'])
print(groups.target)
print(np.unique(groups.target))
print(groups.data[5])
print(groups.target[5])
print(groups.target_names[groups.target[5]])
print(len(groups.data[0]))