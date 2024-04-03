import numpy as np
import pandas as pd
from pprint import pprint
eps = np.finfo(float).eps

def findEntropy(df):
    target = df.keys()[-1]
    values = df[target].unique()
    entropy = 0
    for value in values:
        fraction = df[target].value_counts()[value]/len(df[target])
        entropy += -fraction * np.log2(fraction)
    return entropy

def findAttrEntropy(df, attr):
    target = df.keys()[-1]
    values = df[target].unique()
    attr_values = df[attr].unique()
    entropy = 0
    for attr_value in attr_values:
        ent_ftr = 0
        for value in values:
            num = len(df[attr] [df[target] == value] [df[attr] == attr_value])
            den = len(df[attr] [df[target] == value])
            fraction = num / (den + eps)
            ent_ftr -= fraction * np.log2(fraction)
        fraction2 = den / len(df)
        entropy += fraction2 * ent_ftr
    return entropy

def findDecider(df):
    info_gain = []
    for key in df.keys()[1:-1]:
        info_gain.append(findEntropy(df) - findAttrEntropy(df, key))
    return df.keys()[1:-1][np.argmax(info_gain)]

def getSubtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)

def buildTree(df, tree=None):
    target = df.keys()[-1]
    node = findDecider(df)
    values = df[node].unique()

    if tree is None:
        tree = {}
        tree[node] = {}
    
    for value in values:
        subtable = getSubtable(df, node, value)
        if len(subtable[target].unique()) == 1:
            tree[node][value] = subtable[target][0]
        else:
            tree[node][value] = buildTree(subtable)
    return tree

def classifySample(sample, tree):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    if attribute in sample:
        value = sample[attribute]
        if value in tree[attribute]:
            return classifySample(sample, tree[attribute][value])
    return 'Unknown'

if __name__ == "__main__":
    df = pd.DataFrame(pd.read_csv('tennisDT.csv'))
    t = buildTree(df)
    pprint(t)
    print(classifySample({'outlook': 'Sunny', 'temperature': 'Mild','humidity': 'High','wind':'Weak'}, t))