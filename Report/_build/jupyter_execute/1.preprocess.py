#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# In[1]:


import pandas as pd
import numpy as np
import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# os.getcwd()


# In[2]:


os.getcwd()


# In[3]:


file_dir=os.getcwd()[:-6]+ 'Data/'


# In[4]:


price_2022 = pd.read_csv(file_dir+'raw/hourly/hourly_data_2022.csv')
price_2023 = pd.read_csv(file_dir+'raw/hourly/hourly_data_2023.csv')
demand_2022 = pd.read_csv(file_dir+'raw/hourly/PUB_Demand_2022.csv')
demand_2023 = pd.read_csv(file_dir+'raw/hourly/PUB_Demand_2023.csv')


# In[5]:


price = pd.concat([price_2022,price_2023]).reset_index().drop(columns={'index'})


# In[6]:


price.head()


# In[7]:


demand = pd.concat([demand_2022,demand_2023]).reset_index().drop(columns={'index'}) 


# In[8]:


demand.head()


# In[9]:


df = pd.merge(price,demand,on=['Date','Hour'],how = 'left')


# In[10]:


df = df[['Date','Hour','HOEP','Hour 1 Predispatch','Hour 2 Predispatch','Hour 3 Predispatch','Market Demand','Ontario Demand']]


# In[11]:


date=pd.to_datetime(df['Date'])
hour=df['Hour']
hour = hour.replace(24,0)
hour=hour.apply(lambda x: datetime.time(x,0))
dt=date.apply(lambda x: str(x.date()))+' '+hour.apply(lambda x: str(x))
dt = pd.to_datetime(dt)
def adjusthour(x):
    if x.hour == 0:
        return x + datetime.timedelta(days=1)
    else:
        return x
dt = dt.apply(adjusthour)

df['Date']=dt
df.drop('Hour',axis=1,inplace=True)


# In[12]:


df.rename(columns={'HOEP':'Price','Hour 1 Predispatch':'1_hour_pred','Hour 2 Predispatch':'2_hour_pred','Hour 3 Predispatch':'3_hour_pred'},inplace=True)


# In[13]:


df['1_lag_pred']=df['1_hour_pred'].shift()


# In[14]:


df['2_lag_pred']=df['2_hour_pred'].shift(2)


# In[15]:


df['3_lag_pred']=df['3_hour_pred'].shift(3)


# In[16]:


df


# In[17]:


df['Price'].min()


# In[18]:


df[df['Price']==-4.43]


# In[19]:


df[df['Price']==689.33]


# In[20]:


df['Price'].mean()


# In[21]:


sns.distplot(df['Price'],bins=40,kde=False, color='red')


# In[22]:


plt.plot(df['Date'], df['Price'])


# In[23]:


plt.plot(df['Date'], df['Market Demand'])


# In[24]:


plt.plot(df['Date'], df['Ontario Demand'])


# In[25]:


df.to_csv(file_dir+'interim/final_data.csv',index=False)


# In[ ]:




