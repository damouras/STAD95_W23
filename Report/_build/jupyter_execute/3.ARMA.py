#!/usr/bin/env python
# coding: utf-8

# # ARMA

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from pylab import rcParams
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import scipy.stats as stats
from pmdarima import auto_arima
import datetime

palettename = "ch:start=.2,rot=-.3"
palette = sns.color_palette(palettename, 10)
palette = palette.as_hex()
sns.set_palette(palette.reverse())
col1 = palette[8]
col2 = palette[5]
col3 = palette[2]
cols = [col1, col2, col3]
palette


# In[2]:


price_2022 = pd.read_csv('../Data/hourly_data_2022.csv')
price_2023 = pd.read_csv('../Data/hourly_data_2023.csv')
price = pd.concat([price_2022,price_2023]).reset_index().drop(columns={'index'})
price.head()


# In[4]:


demand_2022 = pd.read_csv('../Data/PUB_Demand_2022.csv')
demand_2023 = pd.read_csv('../Data/PUB_Demand_2023.csv')
demand = pd.concat([demand_2022,demand_2023]).reset_index().drop(columns={'index'})
demand.head()


# In[9]:


df = pd.merge(price,demand,on=['Date','Hour'],how = 'left')
df = df[['Date','Hour','HOEP','Hour 1 Predispatch','Hour 2 Predispatch','Hour 3 Predispatch','Market Demand','Ontario Demand']]
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
df.rename(columns={'Date':'date','HOEP':'price','Hour 1 Predispatch':'1_hour_pred','Hour 2 Predispatch':'2_hour_pred','Hour 3 Predispatch':'3_hour_pred', 'Market Demand':'market_demand', 'Ontario Demand':'ontario_demand'},inplace=True)
df = df.set_index('date')
df


# In[21]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})


# In[17]:


df[['market_demand', 'ontario_demand']].plot();


# In[12]:


plt.figure()
plt.subplot(211)
plot_acf(df.ontario_demand, ax=plt.gca(), lags = 30)
plt.subplot(212)
plot_pacf(df.ontario_demand, ax=plt.gca(), lags = 30, method = 'ywm')
plt.show()


# In[65]:


def loadfileCombine(startyear = 2012, stopyear = 2022):
    
    
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
    
    # merging the two files
    
    data = pd.DataFrame(datafile)
    
    data = data[["timestamp","Date","load","Year","Month","Hour"]]
    
    # save to csv
    data.to_csv("loadDemand.csv",index=False) #, index=False
    return data


# In[66]:


loadDemand=loadfileCombine()
loadDemand.head()


# In[67]:


df = loadDemand[['timestamp', 'load']].set_index("timestamp")
y = df.load
y


# In[68]:


train = y[:int(0.75*(len(y)))]
valid = y[int(0.75*(len(y))):]
train.plot()
valid.plot();


# In[71]:


model = auto_arima(train, start_p=0, start_q=0, test="adf", trace=True, seasonal=24,d= None, max_d=4, max_p=4)
model.fit(train)


# In[73]:


model


# In[72]:


start_index = valid.index.min()
end_index = valid.index.max()

pred = model.predict()
pred = model.predict(n_periods=len(valid))
pred = pd.DataFrame(pred,index = valid.index,columns=['Prediction'])

forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

# Plot the predictions for validation set
plt.plot(y, label='Train')
plt.plot(forecast, label='Prediction')
plt.show()

