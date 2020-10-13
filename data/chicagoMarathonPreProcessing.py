#%%
import pandas as pandas
from pandas import DataFRame
#%%
#%%
chicagoDF: DataFrame = pd.read_csv("chicago_marathon_2018.csv")
#%%
# How to get the country out of the name field
# Use regex to seperate the country from the name

# Create a country column
chicagoDF['country'] = chicagoDF['name'].str.extract('|((.{3})|)')

#%%

#How to get distinct types