#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Project 2: `make_moons` dataset
# 
# 

# In[1]:


from sklearn.datasets import make_moons

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)


# In[2]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[3]:


from sklearn.model_selection import GridSearchCV

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)


# In[1]:


from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
iris = load_iris()
pipe = Pipeline(steps=[
   ('select', SelectKBest(k=2)),
   ('clf', neibo())])
pipe.fit(iris.data, iris.target)

pipe[:-1].get_feature_names_out()


# So, what are the transformers and estimators? — Transformers are the objects that implement fit() and transform() methods. And estimators are the object that implements fit() and predict() methods.
# 
# https://pythonsimplified.com/what-is-a-scikit-learn-pipeline/
# 
# In layman’s terms, as the name itself says, transformers apply transformations to the data. And estimators, as you would have guessed, are nothing but ML models. 
