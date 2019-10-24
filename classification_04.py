#!/usr/bin/env python
# coding: utf-8

# ## IMPORT MNIST DATASET

# In[1]:


from sklearn.datasets import fetch_openml


# In[2]:


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)


# In[3]:


y


# In[4]:


X.shape


# In[5]:


y.shape


# ## PLOT A SINGLE DIGIT FROM FEATURE 

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import matplotlib
import matplotlib.pyplot as plt


# In[8]:


a_digit = X[36000]
a_digit_image = a_digit.reshape(28,28)


# In[9]:


plt.imshow(a_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")
plt.axis("off")
plt.show()


# In[10]:


y[36000]


# ## SPLITTING DATA INTO TEST SET AND TRAINING SET

# In[11]:


X_train,X_test,y_train,y_test = X[:60000],X[60000:],y[:60000],y[60000:]


# In[ ]:





# ## SHUFFLE THE DATA SET

# In[12]:


import numpy as np


# In[13]:


shuffle_index = np.random.permutation(60000)
X_train,y_train = X_train[shuffle_index], y_train[shuffle_index]


# In[14]:


y_train = y_train.astype(np.int8)


# In[ ]:





# ## TRAINING BINARY CLASSIFIER

# In[15]:


y_train_9 = (y_train == 9)


# In[16]:


y_test_9 = (y_test == 9)


# In[17]:


from sklearn.linear_model import SGDClassifier


# In[18]:


sgd_classifier = SGDClassifier(random_state = 42)


# In[19]:


sgd_classifier.fit(X_train,y_train_9)


# In[20]:


np.unique(y_train_9)


# ## PREDICT

# In[21]:


sgd_classifier.predict([a_digit])


# ## EVALUATE MODEL

# In[22]:


from sklearn.model_selection import cross_val_score


# In[23]:


cross_val_score(sgd_classifier, X_train,y_train_9, cv = 3 , scoring = "accuracy")


# In[24]:


# Just a dumb  classifier
from sklearn.base import BaseEstimator


# In[25]:


class Never9Classifier(BaseEstimator):
    def fit(self, X , y=None):
        pass
    
    def predict(self,X):
        return np.zeros((len(X) , 1), dtype = bool)


# In[26]:


never9_classifier = Never9Classifier()


# In[27]:


cross_val_score(never9_classifier,X_train,y_train_9, cv = 3 , scoring = "accuracy")


# ## CONFUSION MATRIX

# In[28]:


from sklearn.model_selection import cross_val_predict


# In[29]:


y_train_pred = cross_val_predict(sgd_classifier, X_train,y_train_9, cv = 3)


# In[30]:


from sklearn.metrics import confusion_matrix


# In[31]:


confusion_matrix(y_train_9,y_train_pred)


# ## PRECISION AND RECALL

# In[32]:


from sklearn.metrics import precision_score , recall_score


# In[33]:


precision_score(y_train_9,y_train_pred)


# In[34]:


recall_score(y_train_9,y_train_pred)


# ## F1 SCORE

# In[35]:


from sklearn.metrics import f1_score


# In[36]:


f1_score(y_train_9,y_train_pred)


# ## SCORES USED TO CALCULATE PREDICTION

# In[37]:


y_scores = sgd_classifier.decision_function([a_digit])


# In[38]:


y_scores


# In[39]:


threshold = 0


# In[40]:


y_a_digit_pred = y_scores > threshold


# In[41]:


y_a_digit_pred


# ## FIND THRESHOLD TO USE

# In[43]:


y_scores = cross_val_predict(sgd_classifier, X_train,y_train_9 , cv = 3 , method = "decision_function")


# In[44]:


# compute precision , recall  for all possible thresholds
from sklearn.metrics import precision_recall_curve


# In[45]:


precisions,recalls , thresholds = precision_recall_curve(y_train_9,y_scores)


# In[47]:


# plot precision and recall as functions of threshold
def plot_precision_recall_vs_threshold(precisions , recalls , thresholds):
    plt.plot(thresholds , precisions[:-1], "b--", label = "Precision")
    plt.plot(thresholds , recalls[:-1], "g-" , label = "Recall")
    plt.xlabel("Threshold")
    plt.legend(loc = "upper left")
    plt.ylim([0,1])


# In[48]:


plot_precision_recall_vs_threshold(precisions,recalls, thresholds)


# In[50]:


plt.plot(recalls , precisions)


# In[52]:


# at 90% precision
y_train_pred_90 = (y_scores > 15000)


# In[53]:


precision_score(y_train_9,y_train_pred_90)


# In[54]:


recall_score(y_train_9,y_train_pred_90)


# In[ ]:




