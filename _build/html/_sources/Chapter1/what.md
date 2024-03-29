# What is Machine Learning?

Machine Learning is the science (and art) of programming computers so they can *learn from data* {cite:p}`Ger2019`.

Here is a slightly more general definition:

```{epigraph}
[Machine Learning is the] field of study that gives computers the ability to learn without being explicitly programmed.

-- Arthur Samuel, 1959
```



This "<mark>without being explicitly programmed to do so</mark>" is the essential difference between Machine Learning and usual computing tasks. The usual way to make a computer do useful work is to have a human programmer write down rules --- a computer program --- to be followed to turn input data into appropriate answers. Machine Learning turns this around: the machine looks at the input data and the expected task outcome, and figures out what the rules should be. A Machine Learning system is *trained* rather than explicitly programmed. It’s presented with many examples relevant to a task, and it finds statistical structure in these examples that eventually allows the system to come up with rules
for automating the task {cite:p}`Cho2021`.

## Types of Machine Learning Systems
There are many different types of Machine Learning systems that it is useful to classify them in braod categories, based on different criteria. These criteria are not exclusive, and you can combine them in any way you like. 

The most popular criterion for Machine Learning classification is the amount and type of supervision they get during training. In this case there are four major types.
```{glossary}
Supervised Learning
    The training set you feed to the algorithm includes the desired solutions. The machines learn from the data to alter the model to get the desired output. The main task for Supervised Learning is classification and regression.

Unsupervised Learning
    In Unsupervised Learning, the data provided doesn't have class information or desired solutions. We just want to dig some information directly from those data themselves. Usually Unsupervised Learning is used for clustering and dimension reduction.

Reinforcement Learning
    In Reinforcement Learning, there is a reward system to measure how well the machine performs the task, and the machine is learning to find the strategy to maximize the rewards. Typical examples here include gaming AI and walking robots.

Semisupervised Learning
    This is actually a combination of Supervised Learning and Unsupervised Learning, that it is usually used to deal with data that are half labelled. 
```

### Tasks for Supervised Learning

As mentioned above, for Supervised Learning, there are two typical types of tasks:

```{glossary}
Classification
    It is the task of predicting a discrete class labels. A typical classification problem is to see an handwritten digit image and recognize it.

Regression
    It is the task of predicting a continuous quantity. A typical regression problem is to predict the house price based on various features of the house.
```
 
There are a lot of other tasks that are not directly covered by these two, but these two are the most classical Supervised Learning tasks.

```{note}
In this course we will mainly focus on **Supervised Classification problems**.
```

### Classification based on complexity
Along with the popularity boost of deep neural network, there comes another classificaiton: shallow learning vs. deep learning. Basically all but deep neural network belongs to shallow learning. Although deep learning can do a lot of fancy stuffs, shallow learning is still very good in many cases. When the performance of a shallow learning model is good enough comparing to that of a deep learning model, people tend to use the shallow learning since it is usually faster, easier to understand and easier to modify.