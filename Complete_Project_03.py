#!/usr/bin/env python
# coding: utf-8

# ## Load Data 

# In[1]:


import  pandas as pd


# In[2]:


def load_housing_data(housing_path):
    return pd.read_csv(housing_path)


# In[3]:


housing_raw = load_housing_data('datasets/housing.csv')


# In[4]:


housing_raw.head()


# In[5]:


housing_raw.info()


# In[6]:


housing_raw["ocean_proximity"].value_counts()


# In[7]:


housing_raw.describe()


# ## Distribution scatterplot

# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[9]:


housing_raw.hist(bins=50 , figsize = (15,10))


# ## Split data into Train and Test set

# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


train_set , test_set = train_test_split(housing_raw,test_size = 0.2 , random_state = 42)


# In[12]:


train_set.head()


# In[13]:


test_set.head()


# In[14]:


import numpy as np


# In[15]:


housing_raw["income_cat"] = np.ceil(housing_raw["median_income"]/1.5)


# In[16]:


housing_raw["income_cat"].where(housing_raw["income_cat"] < 5 , 5.0 , inplace = True)


# In[17]:


housing_raw["income_cat"].unique()


# ## Stratified sampling based on income category

# In[18]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[19]:


split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2 , random_state = 42)


# In[20]:


split


# In[21]:


for train_index , test_index  in split.split(housing_raw , housing_raw["income_cat"]):
    strat_train_set = housing_raw.loc[train_index]
    strat_test_set = housing_raw.loc[test_index]


# In[22]:


#Income category proportions in the full dataset
housing_raw["income_cat"].value_counts() / len(housing_raw)


# In[23]:


# Income category proportions without stratification
train_set["income_cat"] = np.ceil(train_set["median_income"]/1.5)
train_set["income_cat"].where(train_set["income_cat"] < 5 , 5.0 , inplace = True)


# In[24]:


train_set["income_cat"].value_counts() / len(train_set)


# In[25]:


# Income category proportions without stratification
strat_train_set["income_cat"] = np.ceil(strat_train_set["median_income"]/1.5)
strat_train_set["income_cat"].where(strat_train_set["income_cat"] < 5 , 5.0 , inplace = True)


# In[26]:


strat_train_set["income_cat"].value_counts() / len(strat_train_set)


# In[27]:


# Selection based on stratified sampling ensures the data is sample correctly by proportion
# for now drop the income_cat column
for set_ in (strat_train_set , strat_test_set):
    set_.drop("income_cat", axis = 1 , inplace = True)


# ## Data exploration

# In[28]:


#create copy of data to play around with
strat_train_copy = strat_train_set.copy()


# In[29]:


# densely populated areas in california
strat_train_copy.plot(kind = "scatter", x = "longitude" , y="latitude" , alpha = 0.1,  figsize = (10,8))


# In[30]:


# visualize housing prices
strat_train_copy.plot(kind ="scatter",x="longitude",y="latitude", s = strat_train_copy["population"]/100 , label = "population",
                     alpha=0.4, figsize = (10,7) , c = "median_house_value", cmap = plt.get_cmap("jet") , colorbar = True)

plt.legend()


# ## Looking for correlations

# In[31]:


corr_matrix = strat_train_copy.corr()


# In[32]:


corr_matrix["median_house_value"].sort_values(ascending = False)


# In[33]:


from pandas.plotting import scatter_matrix


# In[34]:


attributes  = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]


# In[35]:


scatter_matrix(strat_train_copy[attributes] , figsize = (12,8))


# In[36]:


strat_train_copy.plot(kind = "scatter" , x = "median_income" , y = "median_house_value" , alpha = 0.1)
# remove rows due to which we see caps at 50000 , 350000...to prevent alogorithms from learning data quirks


# ## Experimenting with attribute combinations

# In[37]:


strat_train_copy["rooms_per_household"] = strat_train_copy["total_rooms"]/strat_train_copy["households"]
strat_train_copy["bedrooms_per_household"] = strat_train_copy["total_bedrooms"]/strat_train_copy["households"]
strat_train_copy["population_per_household"] = strat_train_copy["population"]/strat_train_copy["households"]


# In[38]:


corr_matrix = strat_train_copy.corr()


# In[39]:


corr_matrix["median_house_value"].sort_values(ascending = False)


# ## Prepare data for machine learning process

# In[40]:


#start with clean data again
housing = strat_train_set.drop("median_house_value",axis=1)


# In[41]:


housing_labels = strat_train_set["median_house_value"].copy()


# ## Data cleaning

# In[42]:


#Filling missing total_bedrooms with median
from sklearn.impute import SimpleImputer


# In[43]:


imputer = SimpleImputer(strategy="median")


# In[44]:


housing_num = housing.drop("ocean_proximity", axis =1)


# In[45]:


imputer.fit(housing_num)


# In[46]:


housing_num.info()


# In[47]:


imputer.statistics_


# In[48]:


housing_num.median().values


# In[49]:


X = imputer.transform(housing_num)


# In[50]:


housing_cleaned = pd.DataFrame(X, columns= housing_num.columns)


# In[51]:


housing_cleaned.info()


# ## Handling Text and Categorical Variables

# In[52]:


housing_cat = housing["ocean_proximity"]


# In[53]:


from sklearn.preprocessing import LabelBinarizer


# In[54]:


encoder = LabelBinarizer(sparse_output = True)


# In[55]:


housing_cat_1hot = encoder.fit_transform(housing_cat)


# In[56]:


housing_cat_1hot = encoder.fit_transform(housing_cat)


# In[57]:


housing_cat_1hot


# ## Custom Transformers

# In[58]:


from sklearn.base import BaseEstimator , TransformerMixin


# In[59]:


rooms_ix , bedrooms_ix , population_ix , household_ix = 3,4,5,6


# In[60]:


class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    
    def __init__(self,add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self , X , y = None):
        return self
    
    def transform(self , X ):
        rooms_per_household = X[: , rooms_ix] / X[:, household_ix]
        population_per_household = X[:,population_ix] / X[:, household_ix]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix] / X[:,rooms_ix]
            return np.c_[X, rooms_per_household,population_per_household,bedrooms_per_room]
        else :
            return np.c_[X, rooms_per_household,population_per_household]


# In[61]:


attrib_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)


# In[62]:


housing_extra_attribs = attrib_adder.transform(housing_cleaned.values)


# In[63]:


housing_extra_attribs[0]


# ## Feature scaling

# In[64]:


from sklearn.pipeline import Pipeline


# In[65]:


from sklearn.preprocessing import StandardScaler, Imputer


# In[66]:


num_pipeline = Pipeline([
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder',CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])


# In[67]:


housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[68]:


housing_num_tr


# In[69]:


class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values


# In[70]:


num_attribs =  list(housing_num)
cat_attribs = ["ocean_proximity"]


# In[71]:


num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer',Imputer(strategy = "median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])


# In[123]:


from sklearn.base import TransformerMixin #gives fit_transform method for free
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)


# In[124]:


cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer',MyLabelBinarizer()),
])


# In[125]:


#joinin pipelines
from sklearn.pipeline import FeatureUnion


# In[126]:


full_pipeline = FeatureUnion(transformer_list = [
    ("num_pipeline", num_pipeline),
    ("cat_pipeline",cat_pipeline),
])


# In[127]:


median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace = True)


# In[131]:


housing_prepared = full_pipeline.fit_transform(housing)


# In[132]:


housing_prepared


# ## Select and Train Model

# ## Selecting and Evaluating on the Training Set

# In[133]:


from sklearn.linear_model import LinearRegression


# In[134]:


linear_reg = LinearRegression()


# In[135]:


linear_reg.fit(housing_prepared, housing_labels)


# In[138]:


#some predictions
sample_data = housing.iloc[:5]
sample_labels = housing_labels.iloc[:5]


# In[140]:


sample_data_prepared = full_pipeline.transform(sample_data)


# In[142]:


print('Predictions:' , linear_reg.predict(sample_data_prepared))


# In[143]:


print('Labels', sample_labels)


# In[144]:


# Measure  RMSE
from sklearn.metrics import mean_squared_error


# In[145]:


housing_predictions = linear_reg.predict(housing_prepared)


# In[146]:


linear_mse = mean_squared_error(housing_labels , housing_predictions)


# In[148]:


linear_rmse = np.sqrt(linear_mse)


# In[149]:


linear_rmse


# In[150]:


# Try Decision Tree
from sklearn.tree import DecisionTreeRegressor


# In[151]:


dtree_reg = DecisionTreeRegressor()
dtree_reg.fit(housing_prepared,housing_labels)


# In[152]:


housing_predictions = dtree_reg.predict(housing_prepared)


# In[153]:


dtree_mse = mean_squared_error(housing_labels , housing_predictions)


# In[155]:


dtree_rmse = np.sqrt(dtree_mse)


# In[156]:


dtree_rmse


# ## Evaluation using cross-validation

# In[157]:


from sklearn.model_selection import cross_val_score


# In[158]:


scores = cross_val_score(dtree_reg, housing_prepared, housing_labels , scoring = "neg_mean_squared_error" , cv = 10)


# In[159]:


scores


# In[160]:


dtree_rmse_scores = np.sqrt(-scores)


# In[161]:


def display_scores(scores):
    print('scores :', scores)
    print('mean :', scores.mean())
    print('standard deviation :', scores.std())


# In[162]:


display_scores(dtree_rmse_scores)


# In[163]:


linear_reg_scores = cross_val_score(linear_reg , housing_prepared , housing_labels , scoring = "neg_mean_squared_error" , cv = 10)


# In[167]:


linear_reg_rmse = np.sqrt(-linear_reg_scores)


# In[168]:


display_scores(linear_reg_rmse)


# In[170]:


#cross validation using randomforestregressor
from sklearn.ensemble import RandomForestRegressor


# In[171]:


rforest_reg = RandomForestRegressor()


# In[172]:


rforest_reg_scores = cross_val_score(rforest_reg , housing_prepared , housing_labels , scoring = "neg_mean_squared_error" , cv = 10)


# In[173]:


rforest_reg_rmse = np.sqrt(-rforest_reg_scores)


# In[174]:


display_scores(rforest_reg_rmse)


# ## Persisting the model

# In[175]:


from sklearn.externals import joblib


# In[176]:


joblib.dump(linear_reg , "linear_reg.pkl")


# In[177]:


# retrieve saved model
linear_reg_from_file = joblib.load('linear_reg.pkl')


# In[178]:


linear_reg_from_file


# ## Fine tuning model

# ## GridSearch 

# In[179]:


from sklearn.model_selection import GridSearchCV


# In[183]:


param_grid = [
    {'n_estimators': [3,10,30] , 'max_features': [2,4,6,8,]} ,
    {'bootstrap': [False] , 'n_estimators': [3,10] , 'max_features': [2,3,4]}
]


# In[184]:


rforest_grid_reg = RandomForestRegressor()


# In[185]:


grid_search = GridSearchCV(rforest_grid_reg, param_grid, cv = 5 , scoring = "neg_mean_squared_error")


# In[186]:


grid_search.fit(housing_prepared, housing_labels)


# In[188]:


grid_search.best_params_


# In[189]:


gridsearch_results = grid_search.cv_results_


# In[190]:


gridsearch_results


# In[191]:


for mean_score , params in zip(gridsearch_results["mean_test_score"] , gridsearch_results["params"]):
    print(np.sqrt(-mean_score), params)


# ## Randomized search

# ## Ensemble methods

# ## Review feature importances

# In[193]:


feature_importances = grid_search.best_estimator_.feature_importances_


# In[194]:


feature_importances


# In[196]:


columns = housing.columns


# In[197]:


columns


# ## Evaluate model on the Test set

# In[198]:


final_model = grid_search.best_estimator_


# In[199]:


final_model


# In[201]:


X_test = strat_test_set.drop("median_house_value", axis =1 )


# In[203]:


y_test = strat_test_set["median_house_value"].copy()


# In[205]:


X_test_prepared = full_pipeline.transform(X_test)


# In[206]:


final_predictions = final_model.predict(X_test_prepared)


# In[207]:


final_mse = mean_squared_error(y_test , final_predictions)


# In[208]:


final_rmse = np.sqrt(final_mse)


# In[210]:


final_rmse


# In[ ]:




