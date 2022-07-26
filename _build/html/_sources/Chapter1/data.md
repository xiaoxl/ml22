# Basic setting for Machine learning problems

```{note}
We by default assume that we are dealing with a **Supervised** **Classification** problem.
```



linear algebra

vectorization

python



## Input data structure
Since we are dealing with Supervised Classification problems, the desired solutions are given. These desired solutions in Classification problems are also called *labels*. The properties that the data are used to describe are called *features*. Both features and labels are usually organized as row vectors. 


````{prf:example} 
The example is extracted from {cite:p}`Har2012`. There are some sample data shown in the following table. We would like to use these information to classify bird species.

```{list-table} Bird species classification based on four features
:header-rows: 1
:name: example-table

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

````


## Output data structure

### Binary Classification Problem
When there are only one class, and all we care about is whether a data point belongs to this class or not, we call this type of problem **binary classification** problem. 

In this case, the desired output for each data point is either $1$ or $0$, where $1$ means "belonging to this class" and $0$ means "not belonging to this class".

If there are two classes, we can still use the idea of binary classification to study the problem. We choose one class as our focus. When the data point belongs to the other class, we can simply say it does belong to the class we choose.

### $0$ and $1$
In many cases the desired output is either $0$ or $1$, while the output of the model is a real number between $0$ and $1$. In this case, the output of the model is interpreted as the probability for the data to be in the specific class. When we use this model, we simply choose the class that has the highest probability and claim that the data is belonging to this class. 

In the binary case, the above method can be stated in another way. We choose a threshold, and treat those whose probability are above the threshold to be in the class, and others not. The default value for the threshold is $0.5$, and in this case the method is just a special case for the previous method. 

### One-hot coding
