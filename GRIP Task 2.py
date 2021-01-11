#!/usr/bin/env python
# coding: utf-8

# # Grip Internship Task 2 Prediction using KMean Clustering
# ### Segmenting the provided data into different clusters
# ## By Sajeev M Nair

# In[17]:


#importing all the required  libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans


# In[18]:


#Loading the Data 
data=pd.read_csv(r'C:\Users\user\Desktop\datascience data\iris.csv')
print(data.head())
print(data.describe())


# In[19]:


#selecting respective columns
da= data.iloc[:,:4].values
da


# In[20]:


# Standardising the values
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
datasc = scaler.fit_transform(da)
datasc


# In[21]:


#Creating an Elbow chart to select the optimal Number of Clusters
wcss=[]# Within cluster sum of squares
for cluster in range(1,20):
    kmeans= KMeans(n_clusters=cluster,init='k-means++')
    kmeans.fit(datasc)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 20), wcss)
plt.title('The Elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS') 
plt.show()
    


# In[22]:


# The number of clusters is 3
kmeans= KMeans(n_clusters=3,init= 'k-means++')
da_kmeans=kmeans.fit_predict(datasc)


# In[23]:


#Visualising the Data into various Clusters
plt.scatter(datasc[da_kmeans == 0, 0], datasc[da_kmeans == 0, 1], 
            s = 50, c = 'yellow', label = 'Iris-setosa')
plt.scatter(datasc[da_kmeans == 1, 0], datasc[da_kmeans == 1, 1], 
            s = 50, c = 'red', label = 'Iris-versicolour')
plt.scatter(datasc[da_kmeans == 2, 0], datasc[da_kmeans == 2, 1],
            s = 50, c = 'green', label = 'Iris-virginica')


plt.legend()


# In[24]:


#Count of datapoints on different Clusters
frame=pd.DataFrame(datasc)
frame['cluster']=da_kmeans
frame['cluster'].value_counts()


# In[ ]:




