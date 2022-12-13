#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


# In[63]:


os.getcwd()


# In[64]:


df = pd.read_csv("Croptype_ML.csv")


# In[65]:


print(df)


# In[66]:


res = df.interpolate()


# In[67]:


#importing the dataset
X = res.iloc[:, :-1].values
y = res.iloc[:, -1].values


# In[68]:


#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[69]:


print(y_test)


# In[70]:


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[71]:


#Training the random forest classification on the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[72]:


#making predictions
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[73]:


#displaying confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

