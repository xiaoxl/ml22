#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Project 1: the `iris` dataset
# 
# We are going to use the Decision Tree model to study the `iris` dataset. This dataset has already studied previously using k-NN. Again we will only use the first two features for visualization purpose.
# 
# Since the dataset will be splitted, we will put `X` and `y` together as a single variable `S`. In this case when we split the dataset by selecting rows, the features and the labels are still paired correctly. 
# 
# We also print the labels and the feature names for our convenience.

# In[1]:


from sklearn.datasets import load_iris
import numpy as np
from assests.codes.dt import gini, split, countlabels

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target
y = y.reshape((y.shape[0],1))
S = np.concatenate([X,y], axis=1)

print(iris.target_names)
print(iris.feature_names)


# Then we apply `split` to the dataset `S`. 

# In[2]:


r = split(S)
if r['split'] is True:
    Gl, Gr = r['sets']
    print(r['pair'])
    print('The left subset\'s Gini impurity is {g:.2f},'.format(g=gini(Gl)),
          ' and its label counts is {d:}'.format(d=countlabels(Gl)))
    print('The right subset\'s Gini impurity is {g:.2f},'.format(g=gini(Gr)),
          ' and its label counts is {d}'.format(d=countlabels(Gr)))


# The results shows that `S` is splitted into two subsets based on the `0`-th feature and the split value is `1.9`. 
# 
# The left subset is already pure since its Gini impurity is `0`. All elements in the left subset is label `0` (which is `setosa`). The right one is mixed since its Gini impurity is `0.5`. Therefore we need to apply `split` again to the right subset.

# In[3]:


r = split(Gr)
if r['split'] is True:
    Grl, Grr = r['sets']
    print(r['pair'])
    print('The left subset\'s Gini impurity is {g:.2f},'.format(g=gini(Grl)),
          ' and its label counts is {d:}'.format(d=countlabels(Grl)))
    print('The right subset\'s Gini impurity is {g:.2f},'.format(g=gini(Grr)),
          ' and its label counts is {d}'.format(d=countlabels(Grr)))


# This time the subset is splitted into two more subsets based on the `1`-st feature and the split value is `1.7`. The total Gini impurity is minimized using this split. 
# 
# The decision we created so far can be described as follows:
# 
# 1. Check the first feature `sepal length (cm)` to see whether it is smaller or equal to `1.9`.
#    1. If it is, classify it as lable `0` which is `setosa`.
#    2. If not, continue to the next stage.
# 2. Check the second feature `sepal width (cm)` to see whether it is smaller or equal to `1.7`. 
#    1. If it is, classify it as label `1` which is `versicolor`.
#    2. If not, classify it as label `2` which is `virginica`.
# 
# 
# 
