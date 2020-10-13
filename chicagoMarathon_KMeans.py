#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DataFrame
from pandas import Timedelta
from datetime import datetime
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


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


# In[9]:


def toSeconds(someTime: Timedelta) -> float:
    return someTime.total_seconds()


# In[10]:


chicagoDF['half'] = chicagoDF['half'].apply(toSeconds)


# In[11]:


chicagoDF['finish'] = chicagoDF['finish'].apply(pd.to_timedelta)
chicagoDF['finish'] = chicagoDF['finish'].apply(toSeconds)


# In[12]:


chicagoDF['division'].value_counts()


# In[13]:


population: DataFrame = chicagoDF[['half', 'finish', 'division', 'country']].copy()
population = population.dropna()    


# In[14]:


population['division'] = LabelEncoder().fit_transform(population['division'])


# In[15]:


population['country'] = LabelEncoder().fit_transform(population['country'])


# In[16]:


scaledPopulation = MinMaxScaler().fit_transform(population)


# In[17]:


print(scaledPopulation)


# In[18]:



inertiaMeasurement: dict = {}

clusters = range(1,20)

for potentialClusterSize in clusters:
    kMeansModel: KMeans = KMeans(n_clusters=potentialClusterSize).fit(scaledPopulation)
    inertiaMeasurement[potentialClusterSize] = kMeansModel.inertia_


# In[19]:


plt.plot(list(inertiaMeasurement.keys()), list(inertiaMeasurement.values()))


# In[20]:


optimalClusterSizeModel: KMeans = KMeans(n_clusters=3).fit(scaledPopulation)


# In[21]:


clusterNumberVector = optimalClusterSizeModel.predict(scaledPopulation)


# In[22]:


predictedLabels: DataFrame = pd.DataFrame(clusterNumberVector, columns= ['ClusterNumberVector'])


# In[23]:


labeledPopulation: DataFrame = chicagoDF.join(predictedLabels, how='inner')


# In[24]:


labeledPopulation.head()


# In[25]:


scaledPopulationDF : DataFrame = pd.DataFrame(scaledPopulation)


# In[26]:


labeledPopulation: DataFrame = scaledPopulationDF.join(predictedLabels, how='inner')


# In[27]:


labeledPopulation.boxplot([0],by=['ClusterNumberVector'])


# In[28]:


labeledPopulation.boxplot([1],by=['ClusterNumberVector'])


# In[29]:


labeledPopulation.boxplot([2],by=['ClusterNumberVector'])


# In[30]:


labeledPopulation.boxplot([3],by=['ClusterNumberVector'])


# In[31]:


from  sklearn.decomposition import PCA


# In[32]:


#PCA - Principle Component Analysis


# In[33]:


# reducing the dimensions
essentialEigenVectors: PCA = PCA(n_components=2)


# In[34]:


essentialEigenVectors.fit(scaledPopulation)


# In[35]:


print(essentialEigenVectors.explained_variance_ratio_)


# In[36]:


transformedPopulation: PCA = essentialEigenVectors.transform(scaledPopulation)


# In[37]:


transformedPopulationDF: DataFrame = pd.DataFrame(transformedPopulation, columns=['pc1', 'pc2'])


# In[38]:


transformedPopulationDF.head()


# In[39]:



inertiaMeasurement: dict = {}

clusters = range(1,20)

for potentialClusterSize in clusters:
    kMeansModel: KMeans = KMeans(n_clusters=potentialClusterSize).fit(transformedPopulation)
    inertiaMeasurement[potentialClusterSize] = kMeansModel.inertia_


# In[40]:


plt.plot(list(inertiaMeasurement.keys()), list(inertiaMeasurement.values()))


# In[41]:


optimalClusterSizeModel: KMeans = KMeans(n_clusters=3).fit(transformedPopulation)


# In[42]:


clusterNumberVector = optimalClusterSizeModel.predict(transformedPopulationDF)


# In[43]:


predictedLabels: DataFrame = pd.DataFrame(clusterNumberVector, columns=['ClusterNumber'])


# In[44]:


transformedPopulationDF['Cluster'] = predictedLabels


# In[47]:


import hvplot.pandas


# In[48]:


transformedPopulationDF.hvplot.scatter(
    x = 'pc1',
    y = 'pc2',
    by = 'Cluster'
)


# In[ ]:




