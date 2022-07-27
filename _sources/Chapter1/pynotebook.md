# Python quick guide

## Python Notebook
We mainly use Python Notebook (.ipynb) to write documents for this course. Currently all main stream Python IDE support Python Notebook. All of them are not entirely identical but the differences are not huge and you may choose any you like.

One of the easiest ways to use Python Notebook is through [Google Colab](https://colab.research.google.com/). The best part about it is that you don't need to worry about installation and configuration in the first place, and you can directly start to code. 

To use Google Colab, all you need is a Google account. After login, you will see the following page. 
![](assests/img/20220726214822.png)  

There are two buttons `+ Code` and `+ Text` under the menu bar. You may choose them according to your needs to write codes or texts. The syntax for texts is Markdown which is a very simple light wighted language. In most cases you may just ignore the syntax and write plain texts. 

![](assests/img/20220726215325.png)  
To write codes, you first click `+ Code` button to start a new code block, and then type any Python codes you like inside. You may either use the triangle button on the left to execute the codes, or click `ctrl + enter`. 

Colab already contains popular packages, so it is totally ok if you would like to play with some simple things or even many materials in this course. However since it is online evironment, it has some limitations, like hardware power is capped even if you are paid users, or any runtime instance will be released after around 24 hours. You should also upload any of your own datasets to it since the platform is online. Therefore it is still recommended to set up a local environment once you get familiar with Python Notebook. We will put some instructions in the Appendix.

## Python fundamentals
We will put some very basic Python commands here for you to warm up. The main reference for this part is {cite:p}`Har2012`.
### Indentation
Python is using indentation to denote code blocks. It is not convienent to write in the first place, but it forces you to write clean, readable code.

By the way, the `if` and `for` block are actually straightforward.

::::{grid}
:gutter: 2

:::{grid-item-card} One!
```{code-block} python
if jj < 3:
    jj = jj 
    print("It is smaller than 3.")
```
:::

:::{grid-item-card} Two!
```{code-block} python
if jj < 3:
    jj = jj
print("It is smaller than 3.")
```
:::
::::

::::{grid}
:gutter: 2

:::{grid-item-card} Three!
```{code-block} python
for i in range(3):
    i = i + 1
    print(i)
```
:::

:::{grid-item-card} Four!
```{code-block} python
for i in range(3):
    i = i + 1
print(i)
```
:::
::::
Please tell the differences between the above codes.


### `list` and `dict`
Here are some very basic usage of lists of dictionaries in Python.
```{code-block} python
newlist = list()
newlist.append(1)
newlist.append('hello')
print(newlist)

newlisttwo = [1, 'hello']
print(newlisttwo)

newdict = dict()
newdict['one'] = 'good'
newdict[1] = 'yes'
print(newdict)

newdicttwo = {'one': 'good', 1: 'yes'}
print(newdicttwo)
```


### Loop through lists
When creating `for` loops we may let Python directly loop through lists. Here is an example. The code is almost self-explained.
```{code-block} python
alist = ['one', 2, 'three', 4]

for item in alist:
    print(item)
```
