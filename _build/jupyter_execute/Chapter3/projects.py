#!/usr/bin/env python
# coding: utf-8

# # Exercises and Projects
# 
# ```{exercise}
# The dataset and its scattering plot is given below.
# 
# 1. Please calculate the Gini impurity of the whole set by hand.
# 2. Please apply CART to create the decision tree by hand.
# 3. Please use the tree you created to classify the following points:
#     - $(0.4, 1.0)$
#     - $(0.6, 1.0)$
#     - $(0.6, 0)$
# ```
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
data = {'x0': [0.22, 0.37, 0.42, 0.45, 0.18, 0.20, 0.21, 0.23, 0.35, 0.58,
               0.60, 0.61, 0.62, 0.65, 0.70, 0.75, 0.82, 0.88, 0.90, 0.92],
        'x1': [0.83, 0.78, 0.65, 0.37, 0.57, 0.45, 0.67, 0.22, 0.43, 0.33,
               0.75, 0.50, 0.21, 0.31, 0.64, 0.70, 0.80, 0.82, 0.61, 0.81],
        'y': ['r', 'r', 'b', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 
              'b', 'b', 'r', 'r', 'b', 'r', 'r', 'r', 'r', 'r']}
df = pd.DataFrame(data)

plt.figure(figsize=(7, 7))
plt.scatter(df['x0'], df['x1'], c=df['y'])
_ = plt.xlim(0, 1)
_ = plt.ylim(0, 1)


# In[ ]:




