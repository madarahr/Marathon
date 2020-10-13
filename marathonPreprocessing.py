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


marathonDF: DataFrame = pd.read_csv('./data/marathon_results_2019.csv')


# In[3]:


marathonDF.head()


# In[4]:


marathonDF.drop(columns=['State', 'Citizen', 'Unnamed: 8', 'Proj Time'],inplace=True)
marathonDF.head()


# In[5]:


# Create a new column called country and extract the country from the end of the names of marathoners using regex
# chicagoDF['country'] = chicagoDF['name'].str.extract('\((.{3})\)')


# In[6]:


marathonDF['Country'].value_counts()


# In[7]:


def binCountries(countryBinName: str) -> str:
    if countryBinName in ['USA', 'MEX', 'GBR', 'CHN', 'CAN']:
        return countryBinName
    else:
        return 'Other'


# In[8]:


marathonDF['Country'] = marathonDF['Country'].apply(binCountries)


# In[ ]:


#chicagoDF['half'].apply(type)


# In[9]:


marathonDF['Official Time'] = marathonDF['Official Time'].apply(pd.to_timedelta)


# In[10]:


marathonDF['Official Time'].apply(type)


# In[11]:


def toSeconds(someTime: Timedelta) -> float:
    return someTime.total_seconds()


# In[12]:


raceTimes = ['5K', '10K', '15K', '20K', 'Half', '25K', '30K', '35K', '40K', 'Pace', 'Official Time']


# In[17]:


for raceTime in raceTimes:   
    marathonDF[raceTime] = marathonDF[raceTime].apply(pd.to_timedelta)
#    print(marathonDF[raceTime].apply(type))
    marathonDF[raceTime] = marathonDF[raceTime].apply(toSeconds)
    


# In[19]:


population: DataFrame = marathonDF[['Country']].copy()


# In[20]:


population['Country'] = LabelEncoder().fit_transform(population['Country'])


# In[22]:


scaledPopulation = MinMaxScaler().fit_transform(population)


# In[23]:


print(scaledPopulation)


# In[ ]:




