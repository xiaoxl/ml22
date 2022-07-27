# Basic setting for Machine learning problems

```{note}
We by default assume that we are dealing with a **Supervised** **Classification** problem.
```

## Input and output data structure
Since we are dealing with Supervised Classification problems, the desired solutions are given. These desired solutions in Classification problems are also called *labels*. The properties that the data are used to describe are called *features*. Both features and labels are usually organized as row vectors. 


````{prf:example} 
The example is extracted from {cite:p}`Har2012`. There are some sample data shown in the following table. We would like to use these information to classify bird species.

```{list-table} Bird species classification based on four features
:header-rows: 1

* - Weight (g)
  - Wingspan (cm)
  - Webbed feet?
  - Back color
  - Species
* - 1000.1
  - 125.0
  - No
  - Brown
  - Buteo jamaicensis
* - 3000.7
  - 200.0
  - No
  - Gray
  - Sagittarius serpentarius
* - 3300.0
  - 220.3
  - No
  - Gray
  - Sagittarius serpentarius
* - 4100.0
  - 136.0
  - Yes
  - Black
  - Gavia immer
* - 3.0
  - 11.0
  - No
  - Green
  - Calothorax lucifer
* - 570.0
  - 75.0
  - No
  - Black
  - Campephilus principalis
```
The first four columns are features, and the last column is the label. The first two features are numeric and can take on decimal values. The third feature is binary that can only be $1$ (Yes) or $0$ (No). The fourth feature is an enumeration over the color palette. You may either treat it as categorical data or numeric data, depending on how you want to build the model and what you want to get out of the data. In this example we will use it as categorical data that we only choose it from a list of colors ($1$ --- Brown, $2$ --- Gray, $3$ --- Black, $4$ --- Green). 

Then we are able to transform the above data into the following form:


```{list-table} Vectorized Bird species data
:header-rows: 1

* - Features
  - Labels
* - $\begin{bmatrix}1001.1 & 125.0 & 0 & 1 \end{bmatrix}$
  - $1$
* - $\begin{bmatrix}3000.7 & 200.0 & 0 & 2 \end{bmatrix}$
  - $2$
* - $\begin{bmatrix}3300.0 & 220.3 & 0 & 2 \end{bmatrix}$
  - $2$
* - $\begin{bmatrix}4100.0 & 136.0 & 1 & 3 \end{bmatrix}$
  - $3$
* - $\begin{bmatrix}3.0 & 11.0 & 0 & 4 \end{bmatrix}$
  - $4$
* - $\begin{bmatrix}570.0 & 75.0 & 0 & 3 \end{bmatrix}$
  - $5$
```

Then the Supervised Learning problem is stated as follows: Given the features and the labels, we would like to find a model that can classify future data.

````


## Evaluate a Machine Learning model
Once the model is built, how do we know that it is good or not? The naive idea is to test the model on some brand new data and check whether it is able to get the desired results. The usual way to achieve it is to split the input dataset into three pieces: *training set*, *validation set* and *test set*.

The model is initially fit on the training set. After the first stage of the training is done, the fitted model is used to predict the responses on the validation set. These two results will tell use many information about the model, and there will be some modifications of the model accordingly. It is possible that the model is retrained.

Finally the test set is a data set that is never used during the above process, and is used to provide an unbiased evaluation of a final model fit on the training set. 

The sizes and strategies for dataset division depends on the problem and data available. It is often recommanded that more training data should be used. The typical distribution of training, validation and test is $(6:3:1)$, $(7:2:1)$ or $(8:1:1)$. Sometimes test set is discarded and only training set and validation set are used. In this case the distribution of training and validation data is usually $(7:3)$, $(8:2)$ or $(9:1)$.


## Workflow in developing a machine learning application

The workflow described below is from {cite:p}`Har2012`.

1. Collect data.
2. Prepare the input data.
3. Analyze the input data.
4. Train the algorithm.
5. Test the algorithm.
6. Use it.

In this course, we will mainly focus on Step 4 as well Step 5. These two steps are where the "core" algorithms lie, depending on the algorithm. We will start from the next Chapter to talk about various Machine Learning algorithms and examples.


<!-- 
## Output data structure

### Binary Classification Problem
When there are only one class, and all we care about is whether a data point belongs to this class or not, we call this type of problem **binary classification** problem. 

In this case, the desired output for each data point is either $1$ or $0$, where $1$ means "belonging to this class" and $0$ means "not belonging to this class".

If there are two classes, we can still use the idea of binary classification to study the problem. We choose one class as our focus. When the data point belongs to the other class, we can simply say it does belong to the class we choose.

### $0$ and $1$
In many cases the desired output is either $0$ or $1$, while the output of the model is a real number between $0$ and $1$. In this case, the output of the model is interpreted as the probability for the data to be in the specific class. When we use this model, we simply choose the class that has the highest probability and claim that the data is belonging to this class. 

In the binary case, the above method can be stated in another way. We choose a threshold, and treat those whose probability are above the threshold to be in the class, and others not. The default value for the threshold is $0.5$, and in this case the method is just a special case for the previous method. 

### One-hot coding -->
