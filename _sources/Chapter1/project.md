# Exercises 
These exercises are from {cite:p}`Klo2021`, {cite:p}`Ger2019` and {cite:p}`Har2012`. 
<!-- 
## Python Notebook

### dd
{cite:p}`Klo2021` -->

## Basic Python 

### Hello world!
Please complete the following tasks.
- Write a `for` loop to print values from 0 to 4.
- Combine two lists `['apple', 'orange']` and `['banana']` using `+`.
- Sort the list `['apple', 'orange', 'banana']` using `sorted()`.


````{admonition} Click to show answers.
:class: dropdown
```{code-block} python
for i in range(5):
    print(i)

newlist = ['apple', 'orange'] + ['banana']

sorted(['apple', 'orange', 'banana'])
```

Please be careful about the last line. `sorted()` doesn't change the original list. It create a new list. There are some Python functions which change the inputed object in-place. Please read documents on all packages you use to get the desired results.
````


### Play with `list`, `dict` and `pandas`.
Please complete the following tasks.
- Create a new dictionary `people` with two keys `name` and `age`. The values are all empty list.
- Add `Tony` to the `name` list in `people`. 
- Add `Harry` to the `name` list in `people`.
- Add number 100 to the `age` list in `people`.
- Add number 10 to the `age` list in `people`.
- Find all the keys of `people` and save them into a list `namelist`.
- Convert the dictionary `people` to a Pandas DataFrame `df`.

````{admonition} Click to show answers.
:class: dropdown
```{code-block} python
import pandas as pd

people = {'name': list(), 'age': list()}
people['name'].append('Tony')
people['name'].append('Harry')
people['age'].append(100)
people['age'].append(10)

namelist = people.keys()

df = pd.DataFrame(people)
```
````



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
