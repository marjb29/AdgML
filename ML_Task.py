#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ## Task 1

# In[2]:


df = pd.read_csv('insurance.csv',index_col=None)
df


# In[3]:


df.info


# In[4]:


from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
df['sex'] = LE.fit_transform(df['sex'])
df['smoker'] = LE.fit_transform(df['smoker'])
df['region'] = LE.fit_transform(df['region'])

df


# ##### Label Encoding Info:
# sex: Male - 1, Female - 0 <br>
# smoker: Yes - 1, No - 0 <br>
# region: southwest - 3, southeast - 2, northwest - 1, northeast - 0

# In[5]:


data = df.iloc[:,:6].values
data


# In[6]:


charges = df.iloc[:,-1].values
charges


# In[7]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[8]:


regr = linear_model.LinearRegression()

regr.fit(data, charges)

y_pred = regr.predict(data)


# In[9]:


print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(charges, y_pred))
print("Coefficient of determination: %.2f" % r2_score(charges, y_pred))


# ## Task 2

# In[10]:


df = pd.read_csv('Fraud.csv',index_col=None)
df


# In[11]:


df.info


# In[12]:


from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
df['type'] = LE.fit_transform(df['type'])
df['nameOrig'] = LE.fit_transform(df['nameOrig'])
df['nameDest'] = LE.fit_transform(df['nameDest'])

df


# In[13]:


data = df.iloc[:,:10].values
data


# In[14]:


label = df.iloc[:,-2].values
label


# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)


# In[16]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=16)

logreg.fit(X_train, y_train)


# In[17]:


y_pred = logreg.predict(X_test)


# In[18]:


from sklearn import metrics

cm = metrics.confusion_matrix(y_test, y_pred)
cm


# In[19]:


from sklearn.metrics import classification_report
target_names = ['not spam', 'spam']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[20]:


acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: "+str(acc))

