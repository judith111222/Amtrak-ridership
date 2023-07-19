#!/usr/bin/env python
# coding: utf-8

# In[27]:


import math
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.formula.api as sm
from statsmodels.tsa import tsatools, stattools
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics import tsaplots


# In[28]:


Amtrak_df = pd.read_csv('Amtrak.csv')
Amtrak_df['Date'] = pd.to_datetime(Amtrak_df.Month, format='%d/%m/%Y')


# In[29]:


ridership_df = pd.read_csv('ridership.csv')

ridership_df = tsatools.add_trend(Ridership, trend='ct')
ridership_lm = sm.ols(formula='Ridership ~ trend', data=ridership_df).fit()
# In[39]:


ax = ridership_ts.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Ridership (in 000s)')
ax.set_ylim(1300, 2300)

plt.show()


# In[26]:


ridership_ts = pd.Series(Amtrak_df.Ridership.values,
index=Amtrak_df.Date,
 name='Ridership')


# In[23]:


ridership_ts.index = pd.DatetimeIndex(ridership_ts.index,
 
freq=ridership_ts.index.inferred_freq)


# In[24]:


ax = ridership_ts.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Ridership (in 000s)')
ax.set_ylim(1300, 2300)


# In[31]:


ridership_ts_3yrs = ridership_ts['1997':'1999']


# In[32]:


ridership_df = tsatools.add_trend(ridership_ts, trend='ctt')


# In[34]:


ridership_lm = sm.ols(formula='Ridership ~ trend + trend_squared',
 data=ridership_df).fit()


# In[41]:


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,6))
ridership_ts_3yrs.plot(ax=axes[0])
ridership_ts.plot(ax=axes[1])
for ax in axes:
 ax.set_xlabel('Time')
 ax.set_ylabel('Ridership (in 000s)')
 ax.set_ylim(1300, 2300)
ridership_lm.predict(ridership_df).plot(ax=axes[1])
plt.show()


# # 16.5   Code for Naive and seasonal naive forecasts in a
# 3-year validation set for Amtrak ridership

# In[43]:


nValid = 36
nTrain = len(ridership_ts) - nValid


# In[44]:


train_ts = ridership_ts[:nTrain]
valid_ts = ridership_ts[nTrain:]


# In[45]:


naive_pred = pd.Series(train_ts[-1], index=valid_ts.index)
last_season = train_ts[-12:]
seasonal_pred = pd.Series(pd.concat([last_season]*5)
[:len(valid_ts)].values,
 index=valid_ts.index)


# # plot forecasts and actual in the training and validation
# sets

# In[46]:


ax = train_ts.plot(color='C0', linewidth=0.75, figsize=
(9,7))
valid_ts.plot(ax=ax, color='C0', linestyle='dashed',
linewidth=0.75)
ax.set_xlim('1990', '2006-6')
ax.set_ylim(1300, 2600)
ax.set_xlabel('Time')
ax.set_ylabel('Ridership (in 000s)')
naive_pred.plot(ax=ax, color='green')
seasonal_pred.plot(ax=ax, color='orange')


# In[57]:


# determine coordinates for drawing the arrows and lines
one_month = pd.Timedelta('31 days')
xtrain = (min(train_ts.index), max(train_ts.index) -
one_month)
xvalid = (min(valid_ts.index) + one_month,
max(valid_ts.index) - one_month)
xfuture = (max(valid_ts.index) + one_month, '2006')
xtv = xtrain[1] + 0.5 * (xvalid[0] - xtrain[1])
xvf = xvalid[1] + 0.5 * (xfuture[0] - xvalid[1])
ax.add_line(plt.Line2D(xtrain, (2450, 2450),
color='black', linewidth=0.5))
ax.add_line(plt.Line2D(xvalid, (2450, 2450),
color='black', linewidth=0.5))
ax.add_line(plt.Line2D(xfuture, (2450, 2450),
color='black', linewidth=0.5))
ax.text('1995', 2500, 'Training')
ax.text('2001-9', 2500, 'Validation')
ax.text('2004-7', 2500, 'Future')
ax.axvline(x=xtv, ymin=0, ymax=1, color='black',
linewidth=0.5)
ax.axvline(x=xvf, ymin=0, ymax=1, color='black',
linewidth=0.5)
plt.show()


# In[58]:


regressionSummary(valid_ts, naive_pred)


# In[59]:


regressionSummary(valid_ts, seasonal_pred)


# In[60]:


regressionSummary(train_ts[1:], train_ts[:-1])


# In[61]:


regressionSummary(train_ts[12:], train_ts[:-12])


# In[ ]:




