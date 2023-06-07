#!/usr/bin/env python
# coding: utf-8

# FAKE NEWS DETECTION PROBLEM

# In[ ]:


#Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string


# In[ ]:


#Importing the dataset
df_fake = pd.read_csv("C:\\Users\\user\\Desktop\\Fake news detection\\Fake.csv")
df_true = pd.read_csv("C:\\Users\\user\\Desktop\\Fake news detection\\True.csv")


# In[ ]:


df_fake.head(10)


# In[ ]:


df_true.head(10)


# In[ ]:


df_fake["class"] = 0
df_true["class"] =1


# In[ ]:


#Shape of the dataset
df_fake.shape , df_true.shape


# In[ ]:


#droping rows for manual testing of the dataset
df_fake_test = df_fake.tail(10)
for i in range(23480 , 23470,-1):
    df_fake.drop([i], axis = 0, inplace=True)
df_true_test = df_true.tail(10)
for i in range(21416 , 21406,-1):
    df_true.drop([i], axis = 0, inplace=True)


# In[ ]:


df_test = pd.concat([df_fake_test , df_true_test], axis=0)
df_test.to_csv("C:\\Users\\user\\Desktop\\Fake news detection\\Test.csv")


# In[ ]:


df_merge = pd.concat([df_fake, df_true], axis=0)
df_merge.head(10)


# In[ ]:


df =df_merge.drop(["title","subject","date"], axis=1)
df.head(10)


# In[ ]:


#Shuffling the data
df = df.sample(frac=1)
df.head(10)


# In[ ]:


# checking for the null values
df.isnull().sum()


# In[ ]:


# Function to remove unnecessary characters
def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '',text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    return text


# In[ ]:


df["text"] = df["text"].apply(word_drop)


# In[ ]:


df.head(10)


# In[ ]:


# Defining dependent and independent variable
x= df["text"]
y = df["class"]


# In[ ]:


#Splitting the dataset into train and test
x_train, x_test,y_train, y_test = train_test_split(x,y, test_size=.25)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


#Vectorizing the data
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# ###### Logisstic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


LR= LogisticRegression()
LR.fit(xv_train, y_train)


# In[ ]:


LR.score(xv_test, y_test)


# In[ ]:


#Classification Report
pred_LR = LR.predict(xv_test)
print(classification_report(y_test, pred_LR))


# ###### Decision Tree Classification

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


DTC = DecisionTreeClassifier()
DTC.fit(xv_train, y_train)


# In[ ]:


DTC.score(xv_test, y_test)


# In[ ]:


#Classification Report
pred_DT = DTC.predict(xv_test)
print(classification_report(y_test, pred_DT))


# ##### Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


GBC = GradientBoostingClassifier()
GBC.fit(xv_train, y_train)


# In[ ]:


GBC.score(xv_test,y_test)


# In[ ]:


#Classification report
pred_GBC = GBC.predict(xv_test)
print(classification_report(y_test, pred_GBC))


# ##### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


RFC= RandomForestClassifier()
RFC.fit(xv_train,y_train)


# In[ ]:


RFC.score(xv_test,y_test)


# In[ ]:


#Classification Report
pred_RFC = RFC.predict(xv_test)
print(classification_report(y_test, pred_RFC))


# ##### Manual Testing

# In[ ]:


def output_label(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "Not a Fake News"
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(word_drop)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DTC.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)
    return print("\n\nLogistic Regression Prediction: {} \nDecision Tree Classifier Prediction: {} \nGradient Boost Classifier Prediction: {} \nRandom Forest Classifier Prediction: {}".format(output_label(pred_LR), output_label(pred_DT), output_label(pred_GBC), output_label(pred_RFC)))


# In[ ]:


news = str(input())
manual_testing(news)


# In[ ]:





# In[ ]:




