# Basic setting for Machine learning problems

```{note}
We mainly focus on **supervised** **classification** problem in this section.
```

linear algebra

vectorization

python



## Input data structure
Input data is organized as row vectors. 



## Output data structure

### Binary Classification Problem
When there are only one class, and all we care about is whether a data point belongs to this class or not, we call this type of problem **binary classification** problem. 

In this case, the desired output for each data point is either $1$ or $0$, where $1$ means "belonging to this class" and $0$ means "not belonging to this class".

If there are two classes, we can still use the idea of binary classification to study the problem. We choose one class as our focus. When the data point belongs to the other class, we can simply say it does belong to the class we choose.

### $0$ and $1$
In many cases the desired output is either $0$ or $1$, while the output of the model is a real number between $0$ and $1$. In this case, the output of the model is interpreted as the probability for the data to be in the specific class. When we use this model, we simply choose the class that has the highest probability and claim that the data is belonging to this class. 

In the binary case, the above method can be stated in another way. We choose a threshold, and treat those whose probability are above the threshold to be in the class, and others not. The default value for the threshold is $0.5$, and in this case the method is just a special case for the previous method. 

### One-hot coding
