# Random Forest

## Voting classifier

Assume that we have several trained classifiers. The easiest way to make a better classifer out of what we already have is to build a voting system. That is, each classifier give its own prediction, and it will be considered as a vote, and finally the highest vote will be the prediction of the system. 


## Basic bagging
One approach to get many estimators is to use the same training algorithm for every predictor and train them on different random subsets of the training set. When sampling is performed with replacement, this method is called *bagging* (short for *bootstrap aggregating*). When sampling is performed without replacement, it is called *pasting*.



`sklearn` provides `BaggingClassifier` to directly perform bagging or pasting. The code is as follows.

```{code-block} python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifer(),
                            n_estimators=500,
                            max_samples=100,
                            bootstrap=True)
```
In the above code, `bag_clf` is a bagging classifier, made of 500 `DecisionTreeClassifer`s, and is trained over subsets of size `100`. The option `bootstrap=True` means that it is bagging. If you would like to use pasting, the option is `bootstrap=False`.

This `bag_clf` also has `.fit()` and `.predict()` methods. It is used the same as our previous classifiers.

## Random Forests
When the classifiers used in a bagging classifier are all Decision Trees, the bagging classifier is called a `random forest`. `sklearn` provide `RandomForestClassifier` class. It is almost the same as `BaggingClassifier` + `DecisionTreeClassifer`.


```{code-block} python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16)
```

