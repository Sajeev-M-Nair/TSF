#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install quandl


# In[7]:


import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[13]:


quandl.ApiConfig.api_key='dk6Z-svt_LzGwnUjw8_V'
df = quandl.get('EOD/AAPL')
df['HL_PCT']=(df['Adj_High']-df['Adj_Low'])*100/df['Adj_Close']
df['PCT_Chan']=(df['Adj_Close']-df['Adj_Open'])*100/df['Adj_Open']
df.head()


# In[15]:


da=df[['Adj_Close','Adj_Volume','PCT_Chan','HL_PCT']]
da.head()


# In[74]:


da.fillna(value={'Adj_Close':-99999})
da.fillna(value={'Adj_Volume':-99999})
da.fillna(value={'PCT_Chan':-99999})
da.fillna(value={'HL_PCT':-99999})


# In[29]:


da.shift(-1)


# In[91]:



data= da['Future Price']
data=data.reset_index()
data


# In[94]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[107]:


train,test=train_test_split(data,test_size=0.20)
X_train = np.array(train.index).reshape(-1, 1)
y_train = train['Future Price']
lr=LinearRegression()
lr.fit(X_train,y_train)


# In[110]:


plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
plt.plot(X_train,lr.predict(X_train),color='blue')
plt.title('Close Price V/S Date')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()


# In[111]:


X_test = np.array(test.index).reshape(-1,1)
y_test= test['Future Price']


# In[112]:


y_pred = lr.predict(X_test)


# In[113]:


from sklearn import metrics
MSE = metrics.mean_squared_error(y_test,y_pred)
MSE


# In[ ]:




