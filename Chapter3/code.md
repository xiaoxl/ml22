# CART Algorithms 

## Ideas
Consider a labeled dataset $S$ with totally $m$ elements. We use a feature $k$ and a threshold $t_k$ to split it into two subsets: $S_l$ with $m_l$ elements and $S_r$ with $m_r$ elements. Then the cost function of this split is

$$
J(k, t_k)=\frac{m_l}mGini(S_l)+\frac{m_r}{m}Gini(S_r).
$$
It is not hard to see that the more pure the two subsets are the lower the cost function is. Therefore our goal is find a split that can minimize the cost function.


```{prf:algorithm} Split the Dataset
**Inputs** Given a labeled dataset $S=\{[features, label]\}$.

**Outputs** A best split $(k, t_k)$.

1. For each feature $k$:
    1. For each value $t$ of the feature:
        1. Split the dataset $S$ into two subsets, one with $k\leq t$ and one with $k>t$.
        2. Compute the cost function $J(k,t)$. 
        3. Compare it with the current smallest cost. If this split has smaller cost, replace the current smallest cost and pair with $(k, t)$.
2. Return the smallest pair $(k,t_k)$.
```

We then use this split algorithm recursively to get the decision tree.


````{prf:algorithm} Classification and Regression Tree, CART
**Inputs** Given a labeled dataset $S=\{[features, label]\}$ and a maximal depth `max_depth`.

**Outputs** A decision tree.

1. Starting from the original dataset $S$. Set the working dataset $G=S$.
2. Consider a dataset $G$. If $Gini(G)\neq0$, split $G$ into $G_l$ and $G_r$ to minimize the cost function. Record the split pair $(k, t_k)$.
3. Now set the working dataset $G=G_l$ and $G=G_r$ respectively, and apply the above two steps to each of them.
4. Repeat the above steps, until `max_depth` is reached.
````
Here are the sample codes.

```{code-block} python
def split(G):
    m = len(G)
    gmini = gini(G)
    pair = None
    if gini(G) != 0:
        numOffeatures = len(G[0]) - 1
        for k in range(numOffeatures):
            for j in range(m):
                t = G[j][k]
                Gl = [x for x in G if x[k]<=t]
                Gr = [x for x in G if x[k]>t]
                gl = gini(Gl)
                gr = gini(Gr)
                ml = len(Gl)
                mr = len(Gr)
                g = gl*ml/m + gr*mr/m
                if g < gmini:
                    gmini = g
                    pair = (k, t)
                    Glm = Gl
                    Grm = Gr
        res = {'split': True,
               'pair': pair,
               'sets': (Glm, Grm)}
    else:
        res = {'split': False,
               'pair': pair,
               'sets': G}
    return res
```