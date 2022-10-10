# `AdaBoost`

This is the first algorithm that successfully implements the boosting idea. `AdaBoost` is short for *Adaptive Boosting*. 

## General process
1. Assign weights to each data point. At the begining we could assign weights equally. 
2. Train a classifier based on the weighted dataset, and use it to predict on the training set. Find out all wrong answers.
4. Adjust the weights, by inceasing the weights of data points that are done wrongly in the previous generation.
5. Train a new classifier using the new weighted dataset. Predict on the training set and record the wrong answers. 
6. Repeat the above process to get many classifiers. The training stops either by hitting $0$ error rate, or after a specific number of rounds.
7. The final results is the weighted total of the results from all classifiers we trained.

Now let us talk about the weights tuning process. Assume there are $m$ data points. Then the inital weights are set to be $\dfrac1m$. There are 2 sets of weights. Let $w^{i}$ be weights of the datapoints of the $i$th-generation. Let $\alpha_i$ be the weights of the $i$th classifier. After training the $i$th classifier, the error rate is denoted by $e_i$. Then we have 

$$
e_i=\frac{\text{the total number of data points that are misclassified by the $i$th classifier}}{\text{the total number of data points}}
$$

and 

$$
\alpha_i=\frac12\ln\left(\dfrac{1-e_i}{e_i}\right).
$$