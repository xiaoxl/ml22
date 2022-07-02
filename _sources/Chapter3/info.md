# A little bit Infomation Theory

Based on the naive idea, we need to use the information thoery to discuss how to divide the dataset. Here we need two concepts: *self-information* and *entropy*.

## Definitions

````{margin}

```{note}
Information is related to surprise. If the probability is 1, the surprise is 0. If the probability is 0, the surprise is $\infty$.
```
````
````{prf:definition} Self-information
Given an event $x$ with probability $p(x)$, the *self-information* is defined as 

$$
I(x)=-\log_b[p(x)].
$$

Different choices of the base $b$ correspond to different units of information.
- If the base is $2$, the unit is a *bit*. 
- If the base is $e$, the unit is a *nat*. 
- If the base is $10$, the unit is a *dit*.
````

````{prf:remark}
Shannon's definition of *self-information* was chosen to meet several axioms:
1. An event with probability 100% is perfectly unsurprising and yields no information.
2. The less probable an event is, the most surprising it is and the more information it yields.
3. If two independent events are measured separately, the total amount of information is the sum of the self-information of the individual events.

It can be proved that there is a unique function of probability that meets these three axioms, up to a scaling factor. The scaling factor behaves as the base $b$ in the above definition.
````







The entropy is supposed to be the expectation value of information. 

````{margin}
```{note}
The entropy can be understood as the average number of questions you have to ask in order to know the outcomes. 
```
````


````{prf:definition} Entropy
Given a discrete random variable $X$, with possible outcomes $x_1,\ldots,x_n$, which occur with probability $p(x_i)$, the *entropy* of $X$ is defined by

$$
H=E[I]=-\sum_{i=1}^np(x_i)I(x_i)=-\sum_{i=1}^np(x_i)\log_bp(x_i).
$$
````



````{prf:example} Ball example
Suppose we have some balls that are divided into three groups. There are two colors: orange and blue. The distributions are recorded as follows:

| Group      | Orange    | Blue    |
| :---:          |    :---:       |  :--:    |
| 1              |  6              |  1        |
| 2              |  1              |  10      |
| 3              |  7             |  7        |

We now randomly pick one ball from these groups. The self-information the event that we randomly pick a specific color is calculated in the following table. The base is chosen to be $2$. 

| Group | The self-info of picking Orange | The self-info of picking Blue |
|:---:|:---:|:---:|
|1|$-\log_2(6/7)\approx0.22$|$-\log_2(1/7)\approx2.81$|
|2|$-\log_2(1/11)\approx 3.46$|$-\log_2(10/11)\approx 0.14$|
|3|$-\log_2(7/14)=1$|$-\log_2(7/14)=1$|

The entropy of is computed as follows:
- Group 1: $(6/7)\times (-\log_2(6/7))+(1/7)\times (-\log_2(1/7))\approx 0.59$ .
- Group 2: $(1/11)\times(-\log_2(1/11))+(10/11)\times(-\log_2(10/11))\approx 0.44$ .
- Group 3: $(7/14)\times(-\log_2(7/14))+(7/14)\times(-\log_2(7/14))=1$ .
````



````{prf:example} Coin example
Toss a coin. The self-info of head is $-\log_2(1/2)=1$. The self-info of tail is $-\log_2(1/2)=1$. Then the entropy is 

$$
(1/2)\times (-\log_2(1/2)) + (1/2)\times (-\log_2(1/2)) =1.
$$
````


````{prf:example} Die example
Throw a die. The self-info of one side is $-\log_2(1/6)\approx2.58$. Then the entropy is 

$$
\sum_{i=1}^6(1/6)\times (-\log_2(1/6))\approx 2.58.
$$
````

## Codes for Self-information and Entropy


````{prf:algorithm} Computing Self-information
**Inputs** Given a dataset $S$ where all data are labeled, and a specific label $L$. $S$ is a list where each item is of the format $[data, label]$.

**Outputs** Compute the self-information (base $2$) of the event of picking an element from $S$ that belongs to the specific label $L$.

1. Compute the size of the whole dataset: $n=size(S)$.
2. Compute the size of the subset of elements that are labeled by $L$: 
  
   $$m=\#\{s\in S\mid label(s)=L\}.$$
3. Return $-\log_2(m/n)$.
````


```{note}
Since the data format is $item = [data, label]$, when using `Python` to code, we can use `item[-1]` to get access the label of `item`.
```


````{prf:algorithm} Computing Entropy
**Inputs** Given a dataset $S$ where all data are labeled.

**Outputs** The entropy $Entropy$ of the labeled dataset $S$.

1. Initialize the label list $labelList$ to be empty. 
2. Go through all elements in $S$ to get the counts of all labels in the dataset:
   1. If the label of the element is new (which means that it is not in the $labelList$), add it to the list, and set its counter to be $1$.
   2. If the label of the element is in the list, add $1$ to the counter.
3. Initialize $Entropy$ to be $0$.
4. Go thourgh each labels in $labelList$:
   1. For each label compute the self-information $I(L)$.
   2. Compute $I(L)\times p(L)$ and add it to $Entropy$.
5. The final $Entropy$ is the entropy of the dataset.
````


```{note}
Although in the algorithm we call $labelList$ a list, when implementing the algorithm, since we also neet to record the counts of each label, it is better to make it into a dictionary, where key is the label, and value is the counts.
```

Here is the codes.

```{code-block} python
import numpy as np

def selfInfo(S, L):
    n = len(S)
    m = len([item for item in S if item[-1] == L])
    return -np.log2(m/n)

def entropy(S):
    labelList = dict()
    for data in S:
        if data[-1] in labelList.keys():
            labelList[data[-1]] = labelList[data[-1]] + 1
        else:
            labelList[data[-1]] = 1
    entropy = 0
    n = len(S)
    for label in labelList.keys():
        entropy = entropy - np.log2(labelList[label]/n) * labelList[label]/n
    return entropy
```

```{note}
In the above code when we compute entropy, we could also use `selfInfo(S, label)` to compute `-np.log2(labelList[label]/n)`. However since we already compute the counts for each label in the previous codes, we don't need to call `selfInfo` to count the labels again.
```

