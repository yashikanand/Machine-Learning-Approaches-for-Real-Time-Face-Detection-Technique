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

def features(name):
    name = name.lower()
    return {
        'first-letter': name[0], 
        'first2-letters': name[0:2],
        'first3-letters': name[0:3],
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }

features = np.vectorize(features)
df_X = features(df_names['name'])
df_y = df_names['sex']

dfX_train, dfX_test, dfy_train, dfy_test = train_test_split(df_X, df_y, test_size=0.33, random_state=42)

dv = DictVectorizer()

dv.fit_transform(dfX_train)

from sklearn.tree import DecisionTreeClassifier
 
dclf = DecisionTreeClassifier()
my_xfeatures =dv.transform(dfX_train)
dclf.fit(my_xfeatures, dfy_train)
print ('Accuracy of train results = ')
print(dclf.score(dv.transform(dfX_train), dfy_train)) 
print ('Accuracy of test results = ')
print(dclf.score(dv.transform(dfX_test), dfy_test))

def genderpredictor1(a):
    test_name1 = [a]
    transform_dv =dv.transform(features(test_name1))
    vector = transform_dv.toarray()
    if dclf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")

genderpredictor1("Yokshu")
t2 = time.time()
print t2-t1