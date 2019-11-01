#!/usr/bin/env python
# coding: utf-8

# ## LOAD MOON DATASET

# In[1]:


from sklearn.datasets import make_moons


# In[2]:


dataset = make_moons(n_samples = 10000 , noise = 0.4)


# In[3]:


dataset


# In[4]:


X = dataset[0]


# In[5]:


y = dataset[1]


# ## SPLIT DATA TRAIN AND TEST SET

# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)


# ## FIND OPTIMAL HYPERPARAMETER FOR DECISIONTREECLASSIFIER

# In[8]:


from sklearn.tree import DecisionTreeClassifier


# In[9]:


from sklearn.model_selection import GridSearchCV


# In[10]:


#parameters = { 'max_leaf_nodes': [10, 15, 20,25,30] , 'max_depth' : [3,5,10,15,20] }


# In[11]:


#classifier = DecisionTreeClassifier()


# In[12]:


#grid = GridSearchCV(estimator = classifier , param_grid = parameters ,cv = 5)


# In[13]:


#grid.fit(X_train,y_train)


# In[14]:


#grid.best_params_


# ## TRAIN ON THE OPTIMAL MAX_DEPTH AND MAX_LEAF_NODES INDENTIFIED

# In[15]:


classifier = DecisionTreeClassifier(max_leaf_nodes = 20 , max_depth = 10)


# In[16]:


classifier.fit(X_train,y_train)


# In[17]:


y_pred = classifier.predict(X_test)


# In[18]:


from sklearn.metrics import confusion_matrix , classification_report


# In[19]:


print(classification_report(y_test,y_pred))


# In[ ]:




