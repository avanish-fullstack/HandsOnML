#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


train = pd.read_csv('datasets/train_housing.csv')


# In[7]:


test = pd.read_csv('datasets/test_housing.csv')


# In[9]:


train  = train.drop(['Loan_ID'], axis = 1)


# In[10]:


train.dtypes


# In[11]:


X = train.drop(['Loan_Status'] , axis =1)


# In[13]:


y = train['Loan_Status']


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train,X_test,y_train , y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)


# In[17]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# In[18]:


numeric_transformer = Pipeline(steps = [
    ('imputer' , SimpleImputer(strategy='median')) ,
    ('scaler', StandardScaler())])


# In[19]:


categorical_transformer =  Pipeline(steps = [
    ('imputer',SimpleImputer(strategy ='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[20]:


numeric_features = train.select_dtypes(include = ['int64' , 'float64']).columns


# In[21]:


numeric_features


# In[22]:


categorical_features = train.select_dtypes(include = ['object']).drop(['Loan_Status'] , axis = 1).columns


# In[23]:


categorical_features


# In[25]:


from sklearn.compose import ColumnTransformer


# In[26]:


preprocessor = ColumnTransformer(transformers = [
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer,categorical_features)
])


# ## COMBINE PREPROCESSOR WITH CLASSIFIER

# In[27]:


from sklearn.ensemble import RandomForestClassifier


# In[28]:


randomForest = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])


# In[30]:


randomForest.fit(X_train,y_train)


# In[31]:


y_pred = randomForest.predict(X_test)


# In[32]:


y_pred


# In[35]:


randomForest.score(X_test,y_test)


# In[ ]:




