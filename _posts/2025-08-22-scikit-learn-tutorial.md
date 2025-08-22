---
layout: post
title: "Sciket Learn tutorial"
author: vuvietduy
categories: [NLP, Tutorial]
featured: false
published: true
image:
toc: true
---

# What is Scikit Learn

Scikit Learn is the most popular machine learning package for Python and has a lot of algorithms built in

You can install it using

```
pip install scikit-learn
```

Let's talk about the basic structure of how to use Scikit Learn

- Every algorithm is exposed in scikit-learn via an `Estimator`
- First you'll import the model, the general form is:
  `from sklearn.family import Model`

For example:

```python
from sklearn.linear_model import LinearRegression
```

Estimator parameters: All the parameters of an estimator can be set when it is instantiatied, and have suitable default values

# Practice

```python
import numpy as np
import pandas as pd
df = pd.read_csv('../TextFiles/smsspamcollection.tsv', sep='\t')
df.head()
df.isnull().sum()
len(df)
df['label'].unique()
df['label'].value_counts()
from sklearn.model_selection import train_test_split
X = df[['length', 'punct']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=True)

X_train.shape
X_test.shape
y_train
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(solver='lbfgs')
lr_model.fit(X_train, y_train)
from sklearn import metrics
predictions = lr_model.predict(X_test)
predictions
df = pd.DataFrame(metrics.confusion_matrix(y_test, predictions), index=['ham', 'spam'], columns=['ham', 'spam'])
df
print(metrics.classification_report(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

predictions = nb_model.predict(X_test)

print(metrics.confusion_matrix(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))
from sklearn.svm import SVC
svc_model = SVC(gamma='auto')

svc_model.fit(X_train, y_train)
predictions = svc_model.predict(X_test)

print(metrics.confusion_matrix(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))
t = X_test[0:1]
test = svc_model.predict(t)
print(t)
print(test)

```
