#!/usr/bin/env python
# coding: utf-8

# In[281]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[282]:


os.getcwd()


# In[283]:


#Change to directory that contains the file
os.chdir(r"E:\Homework")


# In[284]:


#Make pandas datframe
df = pd.read_csv("Croptype_ML.csv")


# In[285]:


print(df)


# In[286]:


#Interpolate to fill missing values
res = df.interpolate()


# In[287]:


#Factorize the column of croptype
factor = pd.factorize(res['CropType'])
res.CropType = factor[0]
definitions = factor[1]
print(res.CropType.head())
print(definitions)


# In[288]:


print(res)


# In[289]:


#importing the dataset
X = res.iloc[:, :-1].values
y = res.iloc[:, -1].values


# In[290]:


print(y)


# In[291]:


#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[292]:


print(y_test)


# In[293]:


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[294]:


#Training the random forest classification on the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[295]:


print(y_test)


# In[296]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[297]:


print(y_pred)


# In[298]:


#making predictions
#y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[299]:


#displaying confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:




