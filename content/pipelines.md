---
Title: Learn Scikit Pipelines in 5 min. The coolest thing since sliced bread! 
Date: 11 July 2020
Author: Thomas Ebermann
Img: iphone.jpg
Template: post
Tags: data,blog
---

Moving away from Jupyter Notebooks to production code can be a daunting task. In fact there is a whole subfield of machine learning just trying to get this right.  There is Amazon Sagemaker which allows you to build, deploy, and Monitor Machine Learning Models with [Amazon infrastructure](https://aws.amazon.com/de/sagemaker/). There is also even a way to deploy scikit learn models directly on Google infrastructure, including all the fancy auto scaling. But today we won't talk about this, but rather discuss how to write scikit learn code that encapsulates the data ingestion, transformation and prediction in one simple pipeline, instead of being independent steps. The technique is called pipelines. 


## Pipelines

To get started with [pipelines in scikit](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) learn you just need to import them. 

```python
from sklearn.pipeline import make_pipeline
```

From here we will create a pipeline that will first load the data, then transform the columns and then do a prediction. This step is fundamentally different than writing a bunch of "cells" in Jupyter notebooks that do each step individually. Why? Well for starters we can run and test the whole pipeline and not just each step. We can also test the whole performance of the pipeline and optimize all hyper parameters that we like instead of just fiddling around with the prediction ones. The final big advantage is that we don't create any data leakage because we always apply the transformations on the new data and can't learn from the training data.  

## 1. Import data

Importing the data works exactly the same way that we would normally do it. For sake of simplicity I just searched for a titanic dataset that was already split up into a test and training dataset. Normally you would need to create your own split here. 


```python
import pandas as pd

cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name']

df = pd.read_csv('https://github.com/shantnu/Titanic-Machine-Learning/blob/master/titanic_train.csv', nrows=10)
X = df[cols]
y = df['Survived']

df_new = pd.read_csv('https://github.com/shantnu/Titanic-Machine-Learning/blob/master/titanic_test.csv', nrows=10)
X_new = df_new[cols]

```

So after the import we have a training and test dataset. Nothing fancy yet. 

## 2. Transform the data

Now comes the interesting part. We would want to one-hot-encode the Columns: Embarked and Sex in order to be able to work with a Logistic Regression or anything that you fancy. We additionally would want to CountVectorize the Names of the passengers, because we think maybe their names will help us to predict the outcome better.  Normally we would take the whole dataset before spliting it up and manually and individually one-hot-encode and count-vectorize it. We would probably used something like pd.get_dummies to do the job.  But here we create a pipeline that does both things in one line of code. Which is a bit like magic. Lets look at some code: 

```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer


one_hot_encoder = OneHotEncoder()
count_vectorizer = CountVectorizer()

column_transformer = make_column_transformer(
    (one_hot_encoder, ['Embarked', 'Sex']),
    (count_vectorizer, 'Name'),
    remainder='passthrough')
```

We first imported and created a standard one-hot-encoder and a count-vectorizer and then we stacked them together into a so called column-transformer. This method basically applies our transformation methods to the individual columns and ignores the other columns. The outcome is an already transformed dataframe. Check it out yourself, I couldn't believe it when I saw it the first time. So whats left to do? Ah the prediction. Well time for the grande finale: 

## 3. Make a prediction 

So the only thing we need to do is to train a model that we can use to predict the data. For the sake of simplicity I've used Logistic Regression here. And we want to wrap everything together into a pipeline that ingests the data and spits out a prediction. Lets look at the code:

```python

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver='liblinear', random_state=1)

pipeline = make_pipeline(column_transformer, log_reg)
pipeline.fit(X, y)
pipeline.predict(X_new)

```

We have just imported the Logistic Regression, initiated it and then made the magic call: make_pipeline(column_transformer, log_reg). This created a pipeline that first does the column transformations and then finishes with a logistic regression. How cool is that?! We just need to fit it and then we can simply use it like any other ML model to predict data. 

But here is the big plus now: In this case we don't need to preprocess the new data first, because we have a pipeline that takes care of this. So we can simply call pipeline.predict(X_new) on the "raw" new data and get a prediction. 

## Conclusion

Now is this code already production ready? Of course not, but the pipeline patterns definitely produces code that feels much closer to what we could call production ready, since we are not constantly fiddling around with data transformations in different places. We simply call a pipeline. In the next blog posts I will show you how we can use this pipeline to search for the best hyper parameters - not only in the machine learning model but also in the transformation steps. And once we are done doing this I will show you how you can easily deploy this to a nice auto scaling REST Google infrastructure. 

