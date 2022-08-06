import numpy as np


def gini(S):
    n = len(S)
    uniqueLabelList = set([s[-1] for s in S])
    g = 1
    for label in uniqueLabelList:
        nl = len([s for s in S if s[-1] == label])
        g = g - (nl/n)**2
    return g


def split(G):
    m = G.shape[0]
    gmini = gini(G)
    pair = None
    if gini(G) != 0:
        numOffeatures = G.shape[1] - 1
        for k in range(numOffeatures):
            for t in G[:, k]:
                Gl = np.array([x for x in G if x[k] <= t])
                Gr = np.array([x for x in G if x[k] > t])
                gl = gini(Gl)
                gr = gini(Gr)
                ml = Gl.shape[0]
                mr = Gr.shape[0]
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


def countlabels(S):
    uniqueLabelList = set([s[-1] for s in S])
    labelCount = dict()
    for label in uniqueLabelList:
        labelCount[label] = 0
    for row in S:
        labelCount[row[-1]] += 1
    return labelCount