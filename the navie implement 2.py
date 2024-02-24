from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import glob
import os
def letters_only(astr):
    return astr.isalpha()
all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

emails,labels =[],[]
file_path='E:/Code/Microsoft Visual Code/python/infosys/spam detection/enron1/spam/'
for filename in glob.glob(os.path.join(file_path,'*.txt')):
    with open(filename,'r',encoding= "ISO-8859-1") as infile:
        emails.append (infile. read())
        labels.append (1)

file_path='E:/Code/Microsoft Visual Code/python/infosys/spam detection/enron1/ham/'
for filename in glob.glob(os.path.join(file_path,'*.txt')): #glob.glo
    with open(filename,'r',encoding="ISO-8859-1") as infile:
        emails.append (infile. read())
        labels.append (0) # labels. appen
def clean_text(docs):
    cleaned_doc=[]
    for doc in docs:
        cleaned_doc.append(" ".join([lemmatizer.lemmatize(word.lower()) for word in doc.split() if letters_only(word) and word not in all_names]))
    return cleaned_doc

clean_email = clean_text(emails)
print(clean_email[0])
cv=CountVectorizer(stop_words="english", max_features=500)
term_docs = cv.fit_transform(clean_email)
print(term_docs[0])