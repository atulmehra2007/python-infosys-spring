from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import glob
import os
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