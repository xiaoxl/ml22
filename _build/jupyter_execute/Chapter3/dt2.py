#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Project 2: `make_moons` dataset
# 
# 

# `sklearn` includes various random sample generators that can be used to build artificial datasets of controlled size and complexity. We are going to use `make_moons` in this section. More details can be found [here](https://scikit-learn.org/stable/datasets/sample_generators.html).
# 
#  `make_moons` generate 2d binary classification datasets that are challenging to certain algorithms (e.g. centroid-based clustering or linear classification), including optional Gaussian noise. `make_moons` produces two interleaving half circles. It is useful for visualization. 
# 
#  Let us explorer the dataset first.

# In[1]:


from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
plt.scatter(x=X[:, 0], y=X[:, 1], c=y)


# Now we are applying `sklearn.DecisionTreeClassifier` to construct the decision tree. The steps are as follows.
# 1. Split the dataset into training data and test data. 
# 2. Construct the pipeline. Since we won't apply any transformers there for this problem, we may just use the classifier `sklearn.DecisionTreeClassifier` directly without really construct the pipeline object.
# 3. Consider the hyperparameter space for grid search. For this problme we choose `min_samples_split` and `max_leaf_nodes` as the hyperparameters we need. We will let `min_samples_split` run through 2 to 5, and `max_leaf_nodes` run through 2 to 50. We will use `grid_search_cv` to find the best hyperparameter for our model. For cross-validation, the number of split is set to be `3` which means that we will run trainning 3 times for each pair of hyperparameters.
# 4. Run `grid_search_cv`. Find the best hyperparameters and the best estimator. Test it on the test set to get the accuracy score.

# In[2]:


# Step 1
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[3]:


# Step 3
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np

params = {'min_samples_split': list(range(2, 5)),
          'max_leaf_nodes': list(range(2, 50))}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                              params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)


# In[4]:


# Step 4
from sklearn.metrics import accuracy_score

clf = grid_search_cv.best_estimator_
print(grid_search_cv.best_params_)
y_pred = clf.predict(X_test)
accuracy_score(y_pred, y_test)


# Now you can see that for this `make_moons` dataset, the best decision tree should have at most `17` leaf nodes and the minimum number of samples required to be at a leaft node is `2`. The fitted decision tree can get 86.95% accuracy on the test set. 
# 
# Now we can plot the decision tree and the decision surface.

# In[5]:


from sklearn import tree
plt.figure(figsize=(15, 15), dpi=300)
tree.plot_tree(clf, filled=True)


# In[6]:


from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    cmap=plt.cm.RdYlBu,
    response_method="predict"
)
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap='gray',
    edgecolor="black",
    s=15,
    alpha=.15)


# Since it is not very clear what the boundary looks like, I will draw the decision surface individually below.

# In[7]:


DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    cmap=plt.cm.RdYlBu,
    response_method="predict"
)

