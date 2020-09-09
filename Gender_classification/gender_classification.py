import pandas as pd
import numpy as np
import time 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
t1 = time.time()    

df = pd.read_csv('/home/yashikanand/project/Gender_classification/names_dataset.csv')

df.isnull().isnull().sum()

df_names = df

df_names.sex.replace({'F':0, 'M':1},inplace=True)

df_names.sex.unique()

Xfeatures = df_names['name']

cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)

cv.get_feature_names()

y = df_names.sex

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
   
def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")

genderpredictor("Yokshu")
t2 = time.time()

print t2-t1
print ('Accuracy of test results = ')
print(clf.score(X_test,y_test))
print ('Accuracy of train results = ')
print(clf.score(X_train,y_train))