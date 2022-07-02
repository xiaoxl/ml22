#!/usr/bin/env python
# coding: utf-8

# # 

# In[1]:


from sklearn.datasets import load_iris
import numpy as np
from assests.codes.dt import gini, split

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target
y = y.reshape((y.shape[0],1))
S = np.concatenate([X,y], axis=1)


# In[2]:


r = split(S)
Gl = r['sets'][0]
Gr = r['sets'][1]


# In[82]:





# In[3]:


r = split(S)


# In[4]:


Gl = r['sets'][0]
Gr = r['sets'][1]
r2 = split(Gr)
r1 = split(Gl)


# In[5]:


y


# In[6]:


X


# In[ ]:




