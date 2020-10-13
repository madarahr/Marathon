#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DataFrame
from pandas import Timedelta
from datetime import datetime
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# In[2]:


chicagoDF: DataFrame = pd.read_csv('./data/chicago_marathon_2018.csv')


# In[3]:


# Create a new column called country and extract the country from the end of the names of marathoners using regex
chicagoDF['country'] = chicagoDF['name'].str.extract('\((.{3})\)')


# In[4]:


chicagoDF['country'].value_counts()


# In[5]:


def binCountries(countryBinName: str) -> str:
    if countryBinName in ['USA', 'MEX', 'GBR', 'CHN', 'CAN']:
        return countryBinName
    else:
        return 'Other'


# In[6]:


chicagoDF['country'] = chicagoDF['country'].apply(binCountries)


# In[7]:


chicagoDF['half'] = chicagoDF['half'].apply(pd.to_timedelta)


# In[8]:


chicagoDF['half'].apply(type)


# In[10]:


def toSeconds(someTime: Timedelta) -> float:
    return someTime.total_seconds()


# In[11]:


chicagoDF['half'] = chicagoDF['half'].apply(toSeconds)


# In[12]:


chicagoDF['finish'] = chicagoDF['finish'].apply(pd.to_timedelta)
chicagoDF['finish'] = chicagoDF['finish'].apply(toSeconds)


# In[13]:


chicagoDF['division'].value_counts()


# In[14]:


population: DataFrame = chicagoDF[['half', 'finish', 'division', 'country']].copy()


# In[15]:


population['division'] = LabelEncoder().fit_transform(population['division'])


# In[16]:


population['country'] = LabelEncoder().fit_transform(population['country'])


# In[17]:


scaledPopulation = MinMaxScaler().fit_transform(population)


# In[18]:


print(scaledPopulation)


# In[ ]:




