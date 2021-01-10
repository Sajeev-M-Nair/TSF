#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np


import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv(r"C:\Users\user\Desktop\datascience data\widya.csv")
df.info()
sns.set(style = 'whitegrid',palette= 'deep',font_scale = 1.1,rc = {"figure.figsize":[8,5]} )
sns.distplot(df['btc_market_price'],norm_hist=False,kde=False,bins=30,hist_kws={"alpha":1}).set(xlabel = "Market Price", ylabel = "Price")


# In[8]:


sns.jointplot(x=df['btc_market_price'],y=df['btc_market_cap'])


# In[26]:


da = df[['btc_market_price','btc_market_cap','btc_miners_revenue','btc_n_transactions','btc_cost_per_transaction','btc_cost_per_transaction_percent','btc_difficulty','btc_hash_rate']]


# In[27]:


market_pric_mean=da.btc_market_price.mean()
da.fillna(value = {'btc_market_price':market_pric_mean})
market_cap_mean=da.btc_market_cap.mean()
da.fillna(value = {'btc_market_cap':market_cap_mean})
btc_miners_revenue_mean=da.btc_miners_revenue.mean()
da.fillna(value = {'btc_miners_revenue':btc_miners_revenue_mean})
btc_n_transactions_mean=da.btc_n_transactions.mean()
da.fillna(value = {'btc_n_transactions':btc_n_transactions_mean})
btc_cost_per_transaction_mean=da.btc_cost_per_transaction.mean()
da.fillna(value = {'btc_cost_per_transaction':btc_cost_per_transaction_mean})
btc_difficulty_mean=da.btc_difficulty.mean()
da.fillna(value = {'btc_difficulty':btc_difficulty_mean})
btc_hash_rate_mean=da.btc_hash_rate.mean()
da.fillna(value = {'btc_hash_rate':btc_hash_rate_mean})
print(da)


# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[93]:


x=da.iloc[:,:1].values
y=da.iloc[:,1].values
lr.fit(x,y)
lr= LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.4,random_state=0)
lr= LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
                    

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,lr.predict(x_train),color='blue')
plt.title('Bitcoin market cap V/S market price')
plt.xlabel('Market price')
plt.ylabel('Market cap')
plt.show()


# In[87]:


from sklearn import metrics
print(metrics.mean_squared_error(y_test,y_pred))


# In[ ]:




