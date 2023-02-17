#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from lxml import etree
from lxml import objectify
import numpy as np


# In[2]:


xml_data = objectify.parse('generation_fuel_type_multiday.xml')  # Parse XML data
root = xml_data.getroot()
lst = []
data = []
cols = []

for i in root.getchildren():
  if str(type(i)) == "<class 'lxml.objectify.ObjectifiedElement'>":
    lst.append(i)
    cols.append(i.attrib['Series'])

for child in lst:
  tmp = child.getchildren()
  arr = []
  for subchild in tmp:
    val = subchild.getchildren()
    arr.append(val[0])
  data.append(arr)

print([len(data[0]), len(data[1]), len(data[2])])

for i in range(len(data)):
  data[i] = data[i][:-5]
  data[i] = list(map(float, data[i]))


df = pd.DataFrame(data).T
cols = ["Supply_" + name for name in cols]
df.columns = cols
df


# In[ ]:


# Hourly Ontario Energy Price (HOEP)
# Market clearing price (MCP)

xml_data = objectify.parse('price_multiday.xml')  # Parse XML data
root = xml_data.getroot()
lst = []
data = []
cols = []

for i in root.getchildren():
  if str(type(i)) == "<class 'lxml.objectify.ObjectifiedElement'>":
    lst.append(i)
    cols.append(i.attrib['Series'])

for child in lst:
  tmp = child.getchildren()
  arr = []
  for subchild in tmp:
    val = subchild.getchildren()
    arr.append(val[0])
  data.append(arr)

price_mcp = data[2]
print([len(data[0]), len(data[1]), len(data[2])])

df2 = pd.DataFrame(data).T
df2_cp = df2.copy()
df2.dropna(axis=0, inplace=True)
df2 = df2.loc[:140]
df2

data = df2.T.values.tolist()

#Weird element found
idx = data[0].index('')

for i in range(len(data)):
  data[i].pop(idx)
  data[i] = list(map(float, data[i]))

df2 = pd.DataFrame(data).T
cols = ["Price_" + name for name in cols]
df2.columns = cols
df2


# In[ ]:


xml_data = objectify.parse('ontario_demand_multiday.xml')  # Parse XML data
root = xml_data.getroot()
lst = []
data = []
cols = []

for i in root.getchildren():
  if str(type(i)) == "<class 'lxml.objectify.ObjectifiedElement'>":
    lst.append(i)
    cols.append(i.attrib['Series'])

for child in lst:
  tmp = child.getchildren()
  arr = []
  for subchild in tmp:
    val = subchild.getchildren()
    arr.append(val[0])
  data.append(arr)

print([len(data[0]), len(data[1]), len(data[2])])
demand = data[0]

df3 = pd.DataFrame(data).T
df3_cp = df3.copy()
df3.dropna(axis=0, inplace=True)
df3 = df3.loc[:140]

data = df3.T.values.tolist()

#Weird element found
idx = data[1].index('')

for i in range(len(data)):
  data[i].pop(idx)
  data[i] = list(map(float, data[i]))

df3 = pd.DataFrame(data).T
cols = ["Demand_" + name for name in cols]
df3.columns = cols
df3


# In[ ]:


# df = pd.concat([df, df2, df3], axis=1)
# df.to_csv('5minutes.csv')


# In[ ]:


1729/145


# In[ ]:


plot_val = []
for i in range(len(df3_cp[0])):
  if str(type(df3_cp[0].values[i])) == "<class 'lxml.objectify.FloatElement'>":
    plot_val.append(float(df3_cp[0].values[i]))
plt.plot(plot_val)


# In[ ]:


len(plot_val)


# In[ ]:


plot_val = []
for i in range(len(df2_cp[2])):
  if str(type(df2_cp[2].values[i])) == "<class 'lxml.objectify.FloatElement'>":
    plot_val.append(float(df2_cp[2].values[i]))
plt.plot(plot_val)


# In[ ]:


len(plot_val)


# In[ ]:




