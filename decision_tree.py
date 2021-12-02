#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
X = pd.read_csv("zoo_data.csv")
X.columns = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize','type']
X.head()


# ##SPLITTING DATASET

# In[2]:



x= X.values[:,0:16]  
y= X.values[:,16] 

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=1)  


# ##USING ENTROPY

# In[3]:


from sklearn.tree import DecisionTreeClassifier,export_graphviz 
classifier= DecisionTreeClassifier(criterion='entropy', random_state=1)
classifier.fit(x_train, y_train)  
y_pred= classifier.predict(x_test) 
print("y_test=",y_test)
print("y_pred=",y_pred)

from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))
print("Accuracy:",accuracy_score(y_test, y_pred))

from six import StringIO  
from IPython.display import Image  
import pydotplus
dot_data = StringIO()
featured_cols=['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone','breathes','venomous','fins','legs','tail','domesic','catsize']
export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names =featured_cols,class_names=['1','2','3','4','5','6','7'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('ZOO_entropy.png')
Image(graph.create_png())


# ##USING GINI INDEX

# In[4]:


classifier= DecisionTreeClassifier()
classifier.fit(x_train, y_train) 
y_pred= classifier.predict(x_test) 
print(y_test)
print(y_pred)
print(confusion_matrix(y_test, y_pred))
print("Accuracy:",accuracy_score(y_test, y_pred))
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names =featured_cols,class_names=['1','2','3','4','5','6','7'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('ZOO_gini.png')
Image(graph.create_png())


# In[ ]:




