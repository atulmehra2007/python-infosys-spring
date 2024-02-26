from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import glob
import os
import numpy as np
from collections import defaultdict
def letters_only(astr):
    return astr.isalpha()
all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

emails,labels =[],[]
file_path='I:/code/Python/INFOSYS/spam detection/enron1/spam/'
for filename in glob.glob(os.path.join(file_path,'*.txt')):
    with open(filename,'r',encoding= "ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(1)

file_path='I:/code/Python/INFOSYS/spam detection/enron1/ham/'
for filename in glob.glob(os.path.join(file_path,'*.txt')): #glob.glo
    with open(filename,'r',encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(0) # labels. appen
def clean_text(docs):
    cleaned_doc=[]
    for doc in docs:
        cleaned_doc.append(" ".join([lemmatizer.lemmatize(word.lower()) for word in doc.split() if letters_only(word) and word not in all_names]))
    return cleaned_doc

clean_email = clean_text(emails)
print((clean_email[0]))

cv=CountVectorizer(stop_words="english", max_features=500)
term_docs = cv.fit_transform(clean_email)
print(term_docs[0])
feature_names = cv.get_feature_names_out()
print(feature_names[0:10])
feature_mapping=cv.vocabulary_
print(feature_mapping)
print(feature_names[56])
def get_label_index(label):
    label_index=defaultdict(list)
    for index,label in enumerate(labels):
        label_index[label].append(index)
    return label_index
label_index=get_label_index(labels)
print(label_index) 

def get_prior(label_index):
    """ compute prior based on training samples
    Args: 
        label_index(grouped sample indices by classs)
    Return:
        dictionary, with class label as key, corresponing prior as the value
    """
    prior={label: len(index) for label , index in label_index.items()}
    total_count=sum(prior.values())
    for label in prior:
        prior[label]/=float(total_count)
    return prior
def get_likelihood(term_document_matrix,label_index,smoothing=0):
    """ compute likelihood based on training samples
    Args: 
            term_document_matrix(sparse matrix)
            label_index(grouped sample indices by classs)
            smoothing (integer, additive Laplace smoothing parameter)
    Returns:
            dictionary, with class as key, corresponding conditional 
            probability P(feature|class) vector as value
    """
    likelihood={}
    for label,index in label_index.items():
        likelihood[label]=term_document_matrix[index,:].sum(axis=0)+smoothing
        likelihood[label]=np.asarray(likelihood[label])[0]
        total_count = likelihood[label].sum()
        likelihood[label]=likelihood[label]/float(total_count)
    return likelihood
smoothing =1
likelihood=get_likelihood(term_docs,label_index,smoothing)
print(likelihood[0])
print(len(likelihood[0]))
print(likelihood[1][56])
print (feature_names[:45])
def get_posteriors(term_document_matrix,prior,likelihood):  
    """ compute posterior of testing samples , based on prior and likelihood 
    Args: 
            term_document_matrix(sparse matrix)
            prior(dictionary, with class label as key, corresponing prior as the value)
            likelihood(dictionary, with class label as key, corresponding conditional probability vector as value)
    Returns:
            dictionary, with class label as key, corresponding posterior as the value
    """
    num_docs=term_document_matrix.shape[0]
    posteriors=[]
    for i in range(num_docs):
        #posterior is proportional to prior
        # #exp(log(prior*likelihood))
        # #exp(log(prior)+log(likelihood))
        posterior={key: np.log(prior_label) for key, prior_label in prior.items()}
        for label, likelihood_label in likelihood.items():
            term_document_vector=term_document_matrix.getrow(i)
            counts=term_document_vector.data
            indices=term_document_vector.indices
            for count, index in zip(counts,indices):
                posterior[label]+=np.log(likelihood_label[index])*count
                #exp(-1000):exp(-999) will cause zero division error
                #however it equates to exp(0):exp(1)
            min_log_posterior=min(posterior.values())
            for label in posterior:
                try:
                    posterior[label]= np.exp(posterior[label]-min_log_posterior)
                except:
                    # if one's log value is excessively large assign it infinity
                    posterior[label]=float('inf')
            # normalize so that all sum up to 1
            sum_posterior=sum(posterior.value())
            for label in posterior:
                if posterior[label]==float('inf'):
                    posterior[label]=1.0
                else:
                    posterior/=sum_posterior
            posteriors.append(posterior.copy())
    return posteriors        



