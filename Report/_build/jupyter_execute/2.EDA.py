#!/usr/bin/env python
# coding: utf-8

# # EDA

# In[1]:


import pandas as pd
import numpy as np
import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


file_dir=os.getcwd()[:-6]+ 'Data/'


# In[3]:


df = pd.read_csv(file_dir+'interim/final_data.csv')


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


df.info()


# In[22]:


df.isna().sum()


# In[14]:


df.describe()


# In[16]:


df[['Price','Market Demand','Ontario Demand']].corr()


# In[20]:


sns.heatmap(df[['Price','Market Demand','Ontario Demand']].corr(),cmap='coolwarm',annot=True)


# In[25]:


df[['Price']].boxplot()


# In[26]:


df[['Market Demand','Ontario Demand']].boxplot()


# In[ ]:




