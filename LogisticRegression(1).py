#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import random
import numpy as np
import math

col_names=['x1','x2','res']
ds=pd.read_csv('Student-University.csv',header=None,names=col_names)

test=ds.values.tolist()

print(len(test))


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn import *
scaler = StandardScaler()
scaler.fit(ds)
#preprocessing.normalize(ds)
print(ds)


# In[3]:


train=[]
for i in range(80):
    r=random.randrange(len(test))
    train.append(test.pop(r))
print(len(train))
print(len(test))


# In[4]:


alpha=0.0001
b0=0.1
b1=0.2
b2=0.3
x1=[i[0] for i in train]
x2=[i[1] for i in train]
y=[i[2] for i in train]


# In[9]:


epoch = 1000
while(epoch>0):
    for i in range(len(x1)):
        pred = 1/(1+np.exp(-(b0+b1*x1[i]+b2*x2[i])))
        b0 = b0+alpha*(y[i]-pred)*pred*(1-pred)
        b1 = b1+alpha*(y[i]-pred)*pred*(1-pred)*x1[i]
        b2 = b2+alpha*(y[i]-pred)*pred*(1-pred)*x2[i]        
    epoch = epoch-1

print(b0,b1,b2)


# In[6]:


x1_test=[i[0] for i in test]
x2_test=[i[1] for i in test]
y_true=[i[2] for i in test]
print(len(x1_test))


# In[10]:


y_pred=[]
for i in range(len(x1_test)):
    pred_test = round((1/(1+np.exp(-(b0+b1*x1_test[i]+b2*x2_test[i])))))
    y_pred.append(pred_test)
print(y_pred)


# In[11]:


from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
X = ds.drop(columns='res',axis=1)
y = ds.res.values
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
model1 = Pipeline([('standardize', scaler),('log_reg', lr)])
model1.fit(X_train, y_train)
logisticRegr.fit(x_train, y_train)
y_test= logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print(score)

