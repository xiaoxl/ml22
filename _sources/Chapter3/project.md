# Exercises and Projects




```{exercise}
The dataset is given by the following plot.

![](assests/img/20220925232516.png)  

1. Please calculate the Gini impurity of the whole set by hand.
2. Please apply CART to create the decision tree by hand.
3. Please use the tree you created to classify the following points:
    - $(0.4, 1.0)$
    - $(0.6, 1.0)$
    - $(0.6, 0)$
```



```{code-block} python
data = {'x0': [0.22, 0.37, 0.42, 0.45, 0.18, 0.20, 0.21, 0.23, 0.35],
        'x1': [0.83, 0.78, 0.65, 0.37, 0.57, 0.45, 0.67, 0.22, 0.43],
        'y': ['r', 'r', 'b', 'r', 'r', 'r', 'r', 'r', 'r', ]}

print(data)
```


```{list-table} Dataset
:header-rows: 1

* - $x_0$
  - $x_1$
  - $y$
* - 0.21
  - 0.82
  - $+$
* - 0.37
  - 0.78
  - $+$

```