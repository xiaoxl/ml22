#!/usr/bin/env python
# coding: utf-8

# # `AdaBoost`
# 
# This is the first algorithm that successfully implements the boosting idea. `AdaBoost` is short for *Adaptive Boosting*. 
# 
# ## Weighted dataset
# We firstly talk about training a Decision Tree on a weighted dataset. The idea is very simple. When building a Decision Tree, we use some method to determine the split. In this course the Gini impurity is used. There are at least two other methods: cross-entropy and misclassified rate. For all three, the count of the elemnts in some classes is the essnetial part. To train the model over the weighted dataset, we just need to upgrade the count of the elements by the weighted count. 
# 
# 
# ````{prf:example}
# Consider the following data:
# 
# ```{list-table} Dataset
# :header-rows: 1
# 
# * - $x_0$
#   - $x_1$
#   - $y$
#   - Weight
# * - 1.0
#   - 2.1
#   - $+$
#   - 0.5
# * - 2.0
#   - 1.1
#   - $+$
#   - 0.125
# * - 1.3
#   - 1.0
#   - $-$
#   - 0.125
# * - 1.0
#   - 1.0
#   - $-$
#   - 0.125
# * - 2.0
#   - 1.0
#   - $+$
#   - 0.125
# ```
# The weighted Gini impurity is
# 
# $$
# \text{WeightedGini}=1-(0.5+0.125+0.125)^2-(0.125+0.125)^2=0.375.
# $$
# 
# You may see that the original Gini impurity is just the weighted Gini impurity with equal weights. Therefore the first tree we get from `AdaBoost` (see below) is the same tree we get from the Decision Tree model in Chpater 3.
# ````
# 
# ## General process
# 
# Here is the rough description of `AdaBoost`.
# 
# 1. Assign weights to each data point. At the begining we could assign weights equally. 
# 2. Train a classifier based on the weighted dataset, and use it to predict on the training set. Find out all wrong answers.
# 3. Adjust the weights, by inceasing the weights of data points that are done wrongly in the previous generation.
# 4. Train a new classifier using the new weighted dataset. Predict on the training set and record the wrong answers. 
# 5. Repeat the above process to get many classifiers. The training stops either by hitting $0$ error rate, or after a specific number of rounds.
# 6. The final results is based on the weighted total votes from all classifiers we trained.
# 
# Now let us talk about the details. Assume there are $N$ data points. Then the inital weights are set to be $\dfrac1N$. There are 2 sets of weights. Let $w^{(i)}$ be weights of the $i$th data points. Let $\alpha_j$ be the weights of the $j$th classifier. After training the $j$th classifier, the error rate is denoted by $e_j$. Then we have 
# 
# $$
# e_j=\frac{\text{the total weights of data points that are misclassified by the $j$th classifier}}{\text{the total weights of data points}}
# $$
# 
# $$
# \alpha_j=\eta\ln\left(\dfrac{1-e_j}{e_j}\right).
# $$
# 
# $$
# w^{(i)}_{\text{new}}\leftarrow\text{normalization} \leftarrow w^{(i)}\leftarrow\begin{cases}w^{(i)}&\text{if the $i$th data is correctly classified,}\\w^{(i)}\exp(\alpha_j)&\text{if the $i$th data is misclassified.}\end{cases}
# $$
# 
# 
# ```{note}
# The first tree is the same tree we get from the regular Decision Tree model. In the rest of the training process, more weights are put on the data that we are wrong in the previous iteration. Therefore the process is the mimic of "learning from mistakes".
# ```
# 
# ```{note}
# The $\eta$ in computing $\alpha_j$ is called the *learning rate*. It is a hyperparameter that will be specified mannually. It does exactly what it appears to do: alter the weights of each classifier. The default is `1.0`. When the number is very small (which is recommended although it can be any positive number), more iterations will be expected. 
# ```
# 
# 
# ## Example 1: the `iris` dataset
# Similar to all previous models, `sklearn` provides `AdaBoostClassifier`. The way to use it is similar to previous models. Note that although we are able to use any classifiers for `AdaBoost`, the most popular choice is Decision Tree with `max_depth=1`. This type of Decision Trees are also called *Decision Stumps*.
# 
# In the following examples, we initialize an `AdaBoostClassifier` with 500 Decision Stumps and `learning_rate=0.5`. 

# In[1]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=1000,
                             learning_rate=.5)


# We will use the `iris` dataset for illustration. The cross_val_score is calculated as follows.

# In[2]:


from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
scores = cross_val_score(ada_clf, X, y, cv=5)
scores.mean()


# ## Example 2: the Horse Colic dataset
# This dataset is from UCI Machine Learning Repository. The data is about whether horses survive if they get a disease called Colic. The dataset is preprocessed as follows. Note that there are a few missing values inside, and we replace them with `0`. 

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data'
df = pd.read_csv(url, delim_whitespace=True, header=None)
df = df.replace("?", np.NaN)
df = df.fillna(0)
X = df.iloc[:, 1:].to_numpy().astype(float)
y = df[0].to_numpy().astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# In[4]:


clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, learning_rate=0.2)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# In[ ]:




