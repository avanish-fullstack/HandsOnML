#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


# In[2]:


dataset = make_moons(n_samples = 100 , noise = 0.4)


# In[3]:


X = dataset[0]
y = dataset[1]


# In[4]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)


# In[5]:


logistic_clf = LogisticRegression()


# In[6]:


random_clf = RandomForestClassifier()


# In[7]:


svm_clf = SVC()


# In[8]:


voting_clf = VotingClassifier(
 estimators = [('lr', logistic_clf) , ('rf' , random_clf) , ('svc', svm_clf)] , 
 voting = 'hard'
)


# In[9]:


voting_clf.fit(X_train,y_train)


# In[10]:


# Accuracy of each score


# In[11]:


from sklearn.metrics import accuracy_score


# In[12]:


for clf in (logistic_clf , random_clf , svm_clf , voting_clf):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    


# In[ ]:




