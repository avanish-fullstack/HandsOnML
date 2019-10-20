#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn


# In[2]:


# Code example
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model


# In[3]:


# Load Data
oecd_bli = pd.read_csv("oecd_bli_2015.csv")


# In[4]:


oecd_bli.head()


# In[5]:


gdp_per_capita = pd.read_csv("gdp_per_capita.csv", delimiter='\t' , encoding = 'latin1' , thousands = ',')


# In[6]:


gdp_per_capita.head()


# In[7]:


def prepare_country_stats(oecd_bli,gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index = "Country" , columns = "Indicator" , values = "Value")
    gdp_per_capita.rename(columns = {"2015": "GDP per capita"} , inplace = True) 
    gdp_per_capita.set_index("Country" , inplace= True)
    full_country_stats = pd.merge(left = oecd_bli , right = gdp_per_capita , left_index = True , right_index = True)
    full_country_stats.sort_values(by = "GDP per capita", inplace = True)
    remove_indices = [0,1,6,8,33,34,35]
    keep_indices = list(set(range(36)) - set(remove_indices) )
    return full_country_stats[["GDP per capita", "Life satisfaction"]].iloc[keep_indices]
    


# In[8]:


gdp_per_capita


# In[9]:


# Prepare data
country_stats = prepare_country_stats(oecd_bli,gdp_per_capita)


# In[10]:


country_stats.info()


# In[11]:


# Prepare the data
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]


# In[12]:


# Visualize the data
country_stats.plot(kind = 'scatter' , x  = "GDP per capita", y = 'Life satisfaction')
plt.show()


# In[13]:


# Select model
from sklearn.linear_model  import LinearRegression


# In[14]:


regressor = LinearRegression()


# In[15]:


regressor.fit(X,y)


# In[20]:


country_stats


# In[18]:


regressor.summary


# In[21]:


pred = regressor.predict([[22587]])


# In[22]:


pred


# In[ ]:




