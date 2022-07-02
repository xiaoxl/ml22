# project

jupyter notebook


## Play with `dict`

### The dataset `iris`
```{code-block} python
from sklearn.datasets import load_iris
iris = load_iris()
```
Please explore this dataset.
- Please get the features for `iris` and save it into `X` as an numpy array.
- What is the meaning of these features?
- Please get the labels for `iris` and save it into `y` as an numpy array.
- What is the meaning of labels?
  
````{admonition} Click to show answers.
:class: dropdown
We first find that `iris` is a dictionary. Then we can look at all the keys by `iris.keys()`. The interesting keys are `data`, `target`, `target_names` and `feature_names`. We can also read the description of the dataset by looking at `DESCR`. 
```{code-block} python
X = iris['data']
print(iris['feature_names'])
y = iris['target']
print(iris['target'])
```
Since the data is already saved as numpy arrays, we don't need to do anything to change its type.
````
