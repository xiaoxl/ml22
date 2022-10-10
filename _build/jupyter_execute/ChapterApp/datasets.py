#!/usr/bin/env python
# coding: utf-8

# # Datasets
# 
# ## the `iris` dataset

# In[1]:


from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# ## The `breast_cancer` dataset

# In[2]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# 
# ## the Horse Colic Dataset
# The data is from the [UCI database](https://archive.ics.uci.edu/ml/datasets/Horse+Colic). The data is loaded as follows. `?` represents missing data.

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data'
df = pd.read_csv(url, delim_whitespace=True, header=None)
df = df.replace("?", np.NaN)


# We will replace the missing by `0`. This part should be modified if you want to improve the performance of your model.

# In[4]:


df = df.fillna(0)
X = df.iloc[:, 1:].to_numpy().astype(float)
y = df[0].to_numpy().astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# 
# ## the Dating dataset
# The data file can be downloaded from {Download}`here<./assests/datasets/datingTestSet2.txt>`. 
# 
# ```{code-block} python
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# 
# df = pd.read_csv('datingTestSet2.txt', sep='\t', header=None)
# X = np.array(df[[0, 1, 2]])
# y = np.array(df[3])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
# ```
# 

# 
# ## The dataset randomly generated
# - `make_moon` dataset
# 
# This is used to generate two interleaving half circles.

# In[5]:


from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# - `make_gaussian_quantiles` dataset
# 
# This is a generated isotropic Gaussian and label samples by quantile. 
# 
# The following code are from [this page](https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py). It is used to generate a relative complex dataset by combining two datesets together.

# In[6]:


from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split

X1, y1 = make_gaussian_quantiles(cov=2.0, n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=300,
                                 n_features=2, n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, -y2 + 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# 
# ## `MNIST` dataset
# 
# There are several versions of the dataset. 
# 
# - `tensorflow` provides the data with the original split.

# In[7]:


import tensorflow.keras as keras
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# - asfasdfasfd

# In[8]:


print(1)


# ## `titanic` dataset
# This is the famuous Kaggle101 dataset. The original data can be download from [the Kaggle page](https://www.kaggle.com/competitions/titanic/data). You may also download the {Download}`training data<./assests/datasets/titanic/train.csv>` and the {Download}`test data<./assests/datasets/titanic/test.csv>` by click the link.
# 
# ```{code-block} python
# import pandas as pd
# dftrain = pd.read_csv('train.csv')
# dftest = pd.read_csv('test.csv')
# ```
# 

# In[9]:


import pandas as pd
dftrain = pd.read_csv('assests/datasets/titanic/train.csv')
dftest = pd.read_csv('assests/datasets/titanic/test.csv')


# The original is a little bit messy with missing values and mix of numeric data and string data. The above code reads the data into a DataFrame. The following code does some basic of preprocess. This part should be modified if you want to improve the performance of your model.
# 
# 1. Only select columns: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`. That is to say, `Name`, `Cabin` and `Embarked` are dropped.
# 2. Fill the missing values in column `Age` and `Fare` by `0`.
# 3. Replace the column `Sex` by the following map: `{'male': 0, 'female': 1}`.

# In[10]:


import pandas as pd
import numpy as np

def getnp(df):
    df['mapSex'] = df['Sex'].map(lambda x: {'male': 0, 'female': 1}[x])
    dfx = df[['Pclass', 'mapSex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
    dfx['Fare'].fillna(0, inplace=True)
    dfx['Age'].fillna(0, inplace=True)
    if 'Survived' in df.columns:
        y = df['Survived'].to_numpy()
    else:
        y = None
    X = dfx.to_numpy()
    return (X, y)

X_train, y_train = getnp(dftrain)
X_test, _ = getnp(dftest)


# For the purpose of submitting to Kaggle, after getting `y_pred`, we could use the following file to prepare for the submission file.
# 
# ```{code-block} python
# def getdf(df, y):
#     df['Survived'] = y
#     return df[['PassengerId', 'Survived']]
# 
# getdf(dftest, y_pred).to_csv('result.csv')
# ```
# 
