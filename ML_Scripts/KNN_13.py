#!/usr/bin/env python
# coding: utf-8

# In[153]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[154]:


os.getcwd()


# In[155]:


#Change to directory that contains the file
os.chdir(r"E:\Homework")


# In[156]:


#Make pandas datframe
df = pd.read_csv("Croptype_ML.csv")


# In[157]:


print(df)


# In[158]:


#Interpolate to fill missing values
res = df.interpolate()


# In[159]:


#Factorize the column of croptype
factor = pd.factorize(res['CropType'])
res.CropType = factor[0]
definitions = factor[1]
print(res.CropType.head())
print(definitions)


# In[160]:


print(res)


# In[161]:


#importing the dataset
X = res.iloc[:, :-1].values
y = res.iloc[:, -1].values


# In[162]:


print(y)


# In[163]:


#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[164]:


print(y_test)


# In[165]:


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[166]:


#Training the random forest classification on the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 10)
classifier.fit(X_train, y_train)


# In[167]:


print(y_test)


# In[168]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[169]:


print(y_pred)


# In[170]:


#making predictions
#y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[171]:


#displaying confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:




