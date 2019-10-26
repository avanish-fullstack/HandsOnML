#!/usr/bin/env python
# coding: utf-8

# In[1]:


import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# In[2]:


train_df = pd.read_csv('datasets/titanic/train.csv')


# In[3]:


test_df = pd.read_csv('datasets/titanic/test.csv')


# In[4]:


train_df.head()


# In[5]:


train_df.info()


# In[6]:


from sklearn.base import BaseEstimator , TransformerMixin


# In[7]:


class FeatureSelector(BaseEstimator , TransformerMixin):
    
    #Class  contructor
    def __init__(self, feature_names):
        self.feature_names = feature_names
    
    #Fit , return self , do nothing here
    def fit(self , x , y= None):
        return self
    
    #Actual transformation happens here
    def transform(self , x , y = None):
        return x[self.feature_names]


# In[8]:


class MissingValuesSubstituter(BaseEstimator , TransformerMixin):
    
    #Class constructor
    def __init__(self):
        pass
    
    #Fit , return self , do nothing here
    def fit(self, x, y = None):        
        return self
    
    #Actual transformation happens here
    def transform(self, x, y  = None):
        x_with_valid_embark = x.dropna(subset=["Embarked"] , axis = 0)
        x_with_valid_embark["Age"] = x_with_valid_embark["Age"].fillna(x_with_valid_embark["Age"].mean())
        x_with_valid_embark["Fare"] = x_with_valid_embark["Fare"].fillna(x_with_valid_embark["Fare"].mean()) 
        x_with_valid_embark["Age*Class"] =  x_with_valid_embark["Age*Class"].fillna(x_with_valid_embark["Age*Class"].mean())
        x_with_valid_embark["Fare_Per_Person"] = x_with_valid_embark["Fare_Per_Person"].fillna(x_with_valid_embark["Fare_Per_Person"].mean()) 
        
        x_with_valid_embark['Fare_Per_Person'].replace([np.inf],0,inplace=True)
        
        
        return x_with_valid_embark 


# In[9]:


class CategoricalDataTransformer(BaseEstimator , TransformerMixin):
    
    #Class constructor
    def __init__(self):
        pass
    
    #Fit, there is nothing to do here. return self
    def fit(self , x , y = None):
        return self
    
    #Transform , here all the action happens
    def transform(self , x , y = None) : 
        x_embarked_class =  pd.get_dummies(x["Embarked"] , drop_first = True)
        x_sex_class = pd.get_dummies(x["Sex"] , drop_first = True)
        x_deck_class = pd.get_dummies(x["Deck"] , drop_first = True)
        x_title_class = pd.get_dummies(x["Title"] , drop_first = True)
        x.drop(["Embarked" , "Sex" , "Deck" , "Title"] , axis = 1 , inplace = True)
        return pd.concat([x , x_embarked_class , x_sex_class,x_deck_class,x_title_class], axis =1)


# In[10]:


title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

class cleanup (BaseEstimator, TransformerMixin) :
    
    #constructor
    def __init_():
        pass
    
    #fit method
    def fit(self, x, y= None):
        return self
    
    #transform method
    def transform (self, x , y = None):
        # Add Title column
        x['Title'] =  x['Name'].map(lambda x: substrings_in_string(x,title_list))
        x['Title'] = x.apply(replace_titles, axis =1)
        
        # Add Family Size column
        x['Family_Size'] = x['SibSp'] + x['Parch']
        
        # Add interaction term ?? Age and Class
        x['Age*Class'] = x['Age'] * x['Pclass']
        
        #Add column Fare per person
        x['Fare_Per_Person'] = x['Fare']/x['Family_Size']
        
        #Turn cabin number into Deck
        x['Deck'] = x['Cabin'].map(lambda x: substrings_in_string(str(x),cabin_list))
        
        return x
        
    


# In[11]:


import string
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print(big_string)
    return np.nan


# In[12]:


def replace_titles(x) :
    title = x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mme'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if(x['Sex'] == 'Male'):
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
        


# In[13]:


cleanuptransformer = cleanup()


# In[14]:


train_cleaned = cleanuptransformer.transform(train_df)


# In[15]:


train_cleaned


# In[16]:


train_cleaned.columns


# In[17]:


feature_names = ['Survived','Pclass', 'Sex', 'Age', 'Fare',  'Embarked', 'Title', 'Family_Size',
       'Age*Class', 'Fare_Per_Person', 'Deck']


# In[18]:


fl = FeatureSelector(feature_names)


# In[19]:


train_updated = fl.transform(train_cleaned)


# In[20]:


train_updated.head()


# In[21]:


train_updated.info()


# In[22]:


mvs = MissingValuesSubstituter()


# In[23]:


train_mvs = mvs.transform(train_updated)


# In[24]:


train_mvs.info()


# In[25]:


train_mvs.head()


# In[26]:


cdt = CategoricalDataTransformer()


# In[27]:


train_cdt  = cdt.transform(train_mvs)


# In[28]:


train_cdt.head()


# In[29]:


train_cdt.info()


# In[30]:


X = train_cdt.drop(["Survived"] , axis = 1)


# In[31]:


y = train_cdt["Survived"]


# In[32]:


from sklearn.preprocessing import StandardScaler


# In[33]:


sc = StandardScaler()


# In[34]:


X.info()


# In[35]:


X_std = sc.fit_transform(X)


# In[36]:


from sklearn.neighbors import KNeighborsClassifier


# In[37]:


classifier = KNeighborsClassifier()


# ## TRY SVC

# In[38]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# In[39]:


classifier = SVC(C = 1000 , gamma = 0.001 , kernel = 'rbf')


# In[40]:


# params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


# In[41]:


# gridsearch = GridSearchCV(estimator = classifier , cv = 10 , param_grid = params )


# In[42]:


# gridsearch.fit(X_std,y)


# In[43]:


# gridsearch.best_params_


# In[44]:


#params =[ { 'n_neighbors' : [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] }]


# In[45]:


#gridsearch = GridSearchCV(estimator = classifier , cv = 10 , param_grid = params )


# In[46]:


# #try multiple classification algorithms
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.model_selection import cross_val_score


# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

# models = []

# def classification_Models(xtrain, ytrain):    
    
#     models.append( ('LR',  LogisticRegression()) )
#     models.append( ('CART',DecisionTreeClassifier()) )
#     models.append( ('KNN', KNeighborsClassifier()) )
#     models.append( ('NB',  GaussianNB()) )
#     models.append( ('LDA',  LinearDiscriminantAnalysis()) )
#     models.append( ('SVM',  SVC()) )

#     modeloutcomes = []
#     modelnames = []
#     for name,model in models:
#         v_results = cross_val_score(model, xtrain, ytrain, cv = 3, 
#                                      scoring='accuracy', n_jobs = -1, verbose = 0)
#         print(name,v_results.mean())
#         modeloutcomes.append(v_results)
#         modelnames.append(name)
        
#     print(modeloutcomes)


# In[47]:


# classification_Models(X_std,y)


# In[48]:


#gridsearch.fit(X_std,y)


# In[49]:


#gridsearch.best_params_


# In[50]:


#classifier = KNeighborsClassifier(n_neighbors = 15)


# In[51]:


X_std.shape


# In[52]:


classifier.fit(X_std,y)


# In[53]:


y_pred = classifier.predict(X_std)


# In[54]:


from sklearn.metrics import classification_report


# In[55]:


print(classification_report(y,y_pred))


# ## PREDICT TEST

# In[56]:


test_df.head()


# In[57]:


test_df.info()


# In[58]:


cleanuptransformer_test = cleanup()


# In[59]:


test_cleaned = cleanuptransformer_test.transform(test_df)


# In[60]:


test_cleaned.shape


# In[61]:


feature_names_test =  ['Pclass', 'Sex', 'Age', 'Fare',  'Embarked', 'Title', 'Family_Size',
       'Age*Class', 'Fare_Per_Person', 'Deck']


# In[62]:


fl = FeatureSelector(feature_names_test)


# In[63]:


test_updated = fl.transform(test_cleaned)


# In[64]:


test_mvs = mvs.transform(test_updated)


# In[65]:


test_cdt  = cdt.transform(test_mvs)


# In[66]:


test_cdt.head()


# In[71]:


test_cdt.info()


# In[68]:


std_test = StandardScaler()


# In[69]:


test_std = std_test.fit_transform(test_cdt)


# In[70]:


y_test_pred =  classifier.predict(test_std)


# In[ ]:


passengerId = test_df["PassengerId"].values


# In[ ]:


final_submission  = pd.DataFrame({'PassengerId': passengerId , 'Survived': y_test_pred})


# In[ ]:


final_submission.to_csv('submission.csv' , index = False)


# In[ ]:




