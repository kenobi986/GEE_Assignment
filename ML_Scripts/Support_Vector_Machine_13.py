#!/usr/bin/env python
# coding: utf-8

# In[137]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[138]:


os.getcwd()


# In[139]:


#Change to directory that contains the file
os.chdir(r"E:\Homework")


# In[140]:


#Make pandas datframe
df = pd.read_csv("Croptype_ML.csv")


# In[141]:


print(df)


# In[142]:


#Interpolate to fill missing values
res = df.interpolate()


# In[143]:


#Factorize the column of croptype
factor = pd.factorize(res['CropType'])
res.CropType = factor[0]
definitions = factor[1]
print(res.CropType.head())
print(definitions)


# In[144]:


print(res)


# In[145]:


#importing the dataset
X = res.iloc[:, :-1].values
y = res.iloc[:, -1].values


# In[146]:


print(y)


# In[147]:


#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[148]:


print(y_test)


# In[149]:


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[150]:


#Training the random forest classification on the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# In[151]:


print(y_test)


# In[152]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[153]:


print(y_pred)


# In[154]:


#making predictions
#y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[155]:


#displaying confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:




