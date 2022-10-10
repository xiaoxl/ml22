# Datasets

## the `iris` dataset

```{code-block} python
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
```



## The `breast_cancer` dataset


```{code-block} python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
```



## the Dating dataset
The data file can be downloaded from {Download}`here<./assests/datasets/datingTestSet2.txt>`. 

```{code-block} python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('datingTestSet2.txt', sep='\t', header=None)
X = np.array(df[[0, 1, 2]])
y = np.array(df[3])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
```



## The dataset randomly generated

- `make_moon` dataset
```{code-block} python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
```

## `MNIST` dataset

There are several versions of the dataset. 

- `tensorflow` provides the data with the original split.
```{code-block} python
import tensorflow.keras as keras
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
```




## `titanic` dataset
This is the famuous Kaggle101 dataset. The original data can be download from [the Kaggle page](https://www.kaggle.com/competitions/titanic/data). You may also download the {Download}`training data<./assests/datasets/titanic/train.csv>` and the {Download}`test data<./assests/datasets/titanic/test.csv>` by click the link.


```{code-block} python
import pandas as pd
dftrain = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')
```


The original is a little bit messy with missing values and mix of numeric data and string data. The above code reads the data into a DataFrame. The following code does some basic of preprocess. This part should be modified if you want to improve the performance of your model.

1. Only select columns: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`. That is to say, `Name`, `Cabin` and `Embarked` are dropped.
2. Fill the missing values in column `Age` and `Fare` by `0`.
3. Replace the column `Sex` by the following map: `{'male': 0, 'female': 1}`.


```{code-block} python
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
```

For the purpose of submitting to Kaggle, after getting `y_pred`, we could use the following file to prepare for the submission file.


```{code-block} python 
def getdf(df, y):
    df['Survived'] = y
    return df[['PassengerId', 'Survived']]

getdf(dftest, y_pred).to_csv('result.csv')
```
