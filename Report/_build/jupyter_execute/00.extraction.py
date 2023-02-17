#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


def loadfileCombine(startyear = 2020, stopyear = 2022):
    
    datafile = pd.DataFrame({'Date':[],'Hour':[],'Ontario Demand':[]}) # initialize empty file
    
    for year in range(startyear, stopyear+1):
        filepath = "http://reports.ieso.ca/public/Demand/PUB_Demand_{}.csv".format(year)
        df =  pd.read_csv(filepath, skiprows= [0,1,2], usecols = lambda x: x in ['Date','Hour','Ontario Demand'],parse_dates=["Date"]) # skip rows 0,1,2
        datafile = pd.concat([datafile,df], axis = 0)
    
    # adding extra attributes
    datafile["Year"] = datafile["Date"].dt.year
    datafile["Month"] = datafile["Date"].dt.month
    datafile["timestamp"] = datafile["Date"].add(pd.to_timedelta(datafile.Hour - 1, unit="h")) # create timestamp variable from Date and Hour
    
    datafile.index = range(len(datafile)) # to have correct index
    datafile = datafile.rename(columns={"Ontario Demand": "load"})
    
    data = pd.DataFrame(datafile)
    
    data = data[["timestamp","Date","load","Year","Month","Hour"]]
    
    # save to csv
    data.to_csv("loadDemand.csv",index=False) #, index=False
    return data


# In[3]:


loadDemand=loadfileCombine()
loadDemand

