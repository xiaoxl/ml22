#!/usr/bin/env python
# coding: utf-8

# # k-NN project: Dating Classification
# 
# The data can be downloaded from [here](https://www.manning.com/downloads/1108) (CH02/datingTestSet2.txt).
# 
# 
# ## Background
# Helen dated several people and rated them using a three-point scale: 3 is best and 1 is worst. She also collected data from all her dates and recorded them in the file attached. These data contains 3 features:
# 
# - Number of frequent flyer miles earned per year
# - Percentage of time spent playing video games
# - Liters of ice cream consumed per week
# 
# We would like to predict her ratings of new dates when we are given the three features. 
# 
# The data contains four columns, while the first column refers to `Mileage`, the second `Gamingtime`, the third `Icecream` and the fourth `Rating`. 
# 
# ## Look at Data
# 
# We first load the data and store it into a DataFrame.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./assests/datasets/datingTestSet2.txt', sep='\t', header=None)
df.head()


# To make it easier to read, we would like to change the name of the columns.

# In[2]:


df = df.rename(columns={0: "Mileage", 1: "Gamingtime", 2: 'Icecream', 3: 'Rating'})
df.head()


# We use `matplotlib.pyplot.scatter` to look at the scattering plots. From the below plots, it seems that `Icecream`.

# In[3]:


fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
scatter1 = ax1.scatter(df['Mileage'], df['Icecream'], 
                     s=10*df['Rating'], c=np.array(df['Rating']))
scatter2 = ax2.scatter(df['Mileage'], df['Gamingtime'], 
                     s=10*df['Rating'], c=np.array(df['Rating']))
scatter3 = ax3.scatter(df['Gamingtime'], df['Icecream'], 
                     s=10*df['Rating'], c=np.array(df['Rating']))

fig.legend(*scatter1.legend_elements(),
           loc="right", title="Rating")
plt.show()


# ## Applying kNN

# In[4]:


from assests.codes.knn import classify_kNN, dataSplit, classify_kNN_test, encodeNorm, decodeNorm

X = np.array(df[['Mileage', 'Gamingtime', 'Icecream']])
y = np.array(df['Rating'])

X_train, y_train, X_test, y_test = dataSplit(X, y, splitrate=0.9)
X_train_norm, parameters = encodeNorm(X_train)
X_test_norm, _ = encodeNorm(X_test, parameters=parameters)



# In[5]:


errorrate = list()
for i in range(1,20):
    errorrate.append(classify_kNN_test(X_test_norm, y_test, X_train_norm, y_train, k=i))
best = np.array(errorrate).argsort()[0]
print(best)
print(errorrate[best])


# In[6]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=10, weights='distance')
neigh.fit(X_train_norm, y_train)
r2 = neigh.predict(X_test_norm)


# In[7]:


r = np.array([classify_kNN(inX, X_train_norm, y_train, k=10) for inX in X_test_norm])


# In[8]:


r-r2


# In[9]:


d = df.to_numpy()


# In[10]:


d


# In[11]:


d.shape


# In[12]:


from sklearn.datasets import load_iris
import numpy as np


# In[13]:


from sklearn.tree import DecisionTreeClassifier


# In[14]:


X = d[:, :3]
y = d[:,-1]


# In[15]:


treeclf = DecisionTreeClassifier(max_depth=3)
treeclf.fit(X, y)


# In[16]:


treeclf


# In[17]:


print(treeclf)


# In[18]:


from sklearn import tree
plt.figure(figsize=(20,20))
tree.plot_tree(treeclf, filled=True)



# In[ ]:





# In[ ]:




