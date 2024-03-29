#!/usr/bin/env python
# coding: utf-8

# # Voting machine
# 
# 
# ## Voting classifier
# 
# Assume that we have several trained classifiers. The easiest way to make a better classifer out of what we already have is to build a voting system. That is, each classifier give its own prediction, and it will be considered as a vote, and finally the highest vote will be the prediction of the system. 
# 
# In `sklearn`, you may use `VotingClassifier`. It works as follows.

# In[1]:


from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

clfs = [('knn', KNeighborsClassifier(n_neighbors=5)),
        ('dt', DecisionTreeClassifier(max_depth=2))]
voting_clf = VotingClassifier(estimators=clfs, voting='hard')


# All classifiers are stored in the list `clfs`, whose elements are tuples. The syntax is very similar to `Pipeline`. What the classifier does is to train all listed classifiers and use the majority vote to predict the class of given test data. 
# If each classifier has one vote, the voting method is `hard`. There is also a `soft` voting method. In this case, every classifiers not only can predict the classes of the given data, but also estimiate the probability of the given data that belongs to certain classes. On coding level, each classifier should have the `predict_proba()` method. In this case, the weight of each vote is determined by the probability computed. In our course we mainly use `hard` voting.
# 
# Let us use `make_moon` as an example. We first load the dataset.

# In[2]:


from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# We would like to apply kNN model. As before, we build a data pipeline `pipe` to first apply `MinMaxScaler` and then `KNeighborsClassifier`.

# In[3]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline(steps=[('scalar', MinMaxScaler()),
                       ('knn', KNeighborsClassifier())])
parameters = {'knn__n_neighbors': list(range(1, 51))}
gs_knn = GridSearchCV(pipe, param_grid=parameters) 
gs_knn.fit(X_train, y_train)
clf_knn = gs_knn.best_estimator_
clf_knn.score(X_test, y_test)


# The resulted accuracy is shown above.

# 
# 
# We then try it with the Decision Tree.

# In[4]:


from sklearn.tree import DecisionTreeClassifier

gs_dt = GridSearchCV(DecisionTreeClassifier(), param_grid={'max_depth': list(range(1, 11)), 'max_leaf_nodes': list(range(10, 30))})
gs_dt.fit(X_train, y_train)
clf_dt = gs_dt.best_estimator_
clf_dt.score(X_test, y_test)


# We would also want to try Logistic regression method. This will be covered in the next Chapter. At current stage we just use the default setting without changing any hyperparameters.

# In[5]:


from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()
clf_lr.fit(X_train, y_train)
clf_lr.score(X_test, y_test)


# Now we use a voting classifier to combine the results.

# In[6]:


from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
clfs = [('knn', KNeighborsClassifier()),
        ('dt', DecisionTreeClassifier()),
        ('lr', LogisticRegression())]
voting_clf = VotingClassifier(estimators=clfs, voting='hard')
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)


# You may compare the results of all these four classifiers. The voting classifier is not guaranteed to be better. It is just a way to form a model.
