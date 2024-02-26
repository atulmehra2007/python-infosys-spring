from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import glob
import os
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

def letters_only(astr):
    return astr.isalpha()

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

emails, labels = [], []
file_path = 'I:/code/Python/INFOSYS/spam detection/enron1/spam/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(1)

file_path = 'I:/code/Python/INFOSYS/spam detection/enron1/ham/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(0)

def clean_text(docs):
    cleaned_doc = []
    for doc in docs:
        cleaned_doc.append(" ".join([lemmatizer.lemmatize(word.lower()) for word in doc.split() if letters_only(word) and word not in all_names]))
    return cleaned_doc

clean_email = clean_text(emails)
print(clean_email[0])

cv = CountVectorizer(stop_words="english", max_features=500)
term_docs = cv.fit_transform(clean_email)
print(term_docs[0])
feature_names = cv.get_feature_names_out()
print(feature_names[0:10])
feature_mapping = cv.vocabulary_
print(feature_mapping)
print(feature_names[56])

def get_label_index(labels):
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index

label_index = get_label_index(labels)
print(label_index) 

def get_prior(label_index):
    prior = {label: len(index) for label, index in label_index.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return prior

def get_likelihood(term_document_matrix, label_index, smoothing=0):
    likelihood = {}
    for label, indices in label_index.items():
        for index in indices:
            #if index >= term_document_matrix.shape[0]:
                #print(f"Warning: Index {index} is out of range for label {label}. Skipping likelihood computation.")
                #continue
            likelihood[label] = term_document_matrix[index, :].sum(axis=0) + smoothing
            likelihood[label] = np.asarray(likelihood[label])[0]
            total_count = likelihood[label].sum()
            likelihood[label] = likelihood[label] / float(total_count)
    return likelihood

def get_posteriors(term_document_matrix, prior, likelihood):
    num_docs = term_document_matrix.shape[0]
    posteriors = []
    for i in range(num_docs):
        posterior = {key: np.log(prior_label) for key, prior_label in prior.items()}
        for label, likelihood_label in likelihood.items():
            term_document_vector = term_document_matrix.getrow(i)
            counts = term_document_vector.data
            indices = term_document_vector.indices
            for count, index in zip(counts, indices):
                posterior[label] += np.log(likelihood_label[index]) * count
        max_log_posterior = max(posterior.values())
        total = 0.0
        for label in posterior:
            try:
                posterior[label] = np.exp(posterior[label] - max_log_posterior)
                total += posterior[label]
            except: # OverflowError:
                #posterior[label] = np.exp(709)
                posterior[label]=float('inf')
                #total += posterior[label]
        sum_posterior=sum(posterior.values())
        for label in posterior:
            if posterior[label]==float('inf'):
                posterior[label]=1.0
            else:
                posterior[label]/=sum_posterior
        posteriors.append(posterior.copy())
        '''for label in posterior:
            posterior[label] /= total
        posteriors.append(posterior.copy())'''

    return posteriors        


X_train, X_test, Y_train, Y_test = train_test_split(clean_email, labels, test_size=0.33, random_state=42)
print(len(X_train), len(X_test))
print(len(Y_train), len(Y_test))

term_docs_train = cv.fit_transform(X_train)
label_index = get_label_index(Y_train)
prior = get_prior(label_index)
smoothing = 1
likelihood = get_likelihood(term_docs_train, label_index, smoothing)
term_docs_test = cv.fit_transform(X_test)
posterior = get_posteriors(term_docs_test, prior, likelihood)
correct = 0.0

for pred, actual in zip(posterior, Y_test):
    if actual == 1:
        if pred[1] >= 0.5:
            correct += 1
        elif pred[0] > 0.5:
            correct += 1
print('The accuracy on {0} testing samples is {1:.1f}%'.format(len(Y_test), (correct / len(Y_test) )* 100))
