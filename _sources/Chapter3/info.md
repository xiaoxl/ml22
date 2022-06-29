# A little bit Infomation Theory

Based on the naive idea, we need to use the information thoery to discuss how to divide the dataset. Here we need two concepts: *self-information* and *entropy*.

````{prf:definition}
:label: my-definition

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



````{prf:example}
There are two type of chickens (orange and blue):
| Group      | Orange    | Blue    |
| :---:          |    :---:       |  :--:    |
| 1              |  6              |  1        |
| 2              |  1              |  10      |
| 3              |  7             |  7        |

The self-information is calculated in the following table. The base is chosen to be $2$. 
| Group | The self-info of picking Orange | The self-info of picking Blue |
|:---:|:---:|:---:|
|1|$-\log_2(6/7)\approx0.22$|$-\log_2(1/7)\approx2.81$|
|2|$-\log_2(1/11)\approx 3.46$|$-\log_2(10/11)\approx 0.14$|
|3|$-\log_2(7/14)=1$|$-\log_2(7/14)=1$|
````
