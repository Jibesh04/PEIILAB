import pandas as pd
import numpy as np
import pprint
# eps is used to avoid division by zero exception and log(0) invalid expression
eps = np.finfo(float).eps

# find entropy for the dataset
def find_entropy(df):
    # assuming the last attribute of dataset to be the target
    target = df.keys()[-1]
    values = df[target].unique()
    # implementing the formula of sum( -(p_i)log_2(p_i))
    entropy = 0
    for value in values:
        fraction = df[target].value_counts()[value] / len(df[target])
        entropy += -fraction * np.log2(fraction)
    return entropy

# find individual attribute's entropy (used in information gain calculation)
def find_entropy_attribute(df, attribute):
    # assuming the last attribute of dataset to be the target
    target = df.keys()[-1]
    # values of the target attribute
    target_values = df[target].unique()
    # values of the considered attribute
    values = df[attribute].unique()
    # implementation of the sum(fraction*entropy(attribute))
    # to be deducted from entropy of dataset to obtain information gain
    entropy_attribute = 0
    for value in values:
        entropy_each_feature = 0
        for target_value in target_values:
            # numerator represents number of instances where the target value is obtained
            num = len(df[attribute] [df[attribute] == value] [df[target] == target_value])
            # denominator represents total number of instances with particular attribute value considered
            den = len(df[attribute] [df[attribute] == value])
            fraction = num / (den + eps)
            entropy_each_feature += -fraction * np.log2(fraction + eps)
        # fraction of number of instances with particular value for attribute w.r.t. total number of instances
        fraction2 = den/len(df)
        entropy_attribute += -fraction2 * entropy_each_feature

    return abs(entropy_attribute)

def find_decider(df):
    # returns the one attribute having highest gain in the passed dataset
    info_gain = []
    for key in df.keys()[1:-1]:
        info_gain.append(find_entropy(df) - find_entropy_attribute(df, key))
    return df.keys()[1:-1][np.argmax(info_gain)]

def get_subtable(df, node, value):
    # returns subtable having particular value of attribute and eliminates the column
    return df[df[node] == value].reset_index(drop=True)


def buildTree(df, tree=None):
    # builds the decision tree
    target = df.keys()[-1]
    # root is the node with highest gain
    node = find_decider(df)
    # different values of the root
    values = np.unique(df[node])

    # initializing empty dictionary to store tree
    if tree is None:
        tree = {}
        tree[node] = {}

    # for each value of the root, further subtree are appended until leaf becomes either of the target values
    for value in values:
        subtable = get_subtable(df, node, value)
        target_value, counts = np.unique(subtable[target], return_counts=True)

        # single target value observed, leaf is added with the value
        if len(counts) == 1:
            tree[node][value] = target_value[0]
        # need to call it recursively to complete the decision tree
        else:
            tree[node][value] = buildTree(subtable)
    return tree

if __name__ == "__main__":
    # reads the dataset
    df = pd.read_csv('tennisDT.csv')
    # decision tree is built
    t = buildTree(df)
    # pprint (pretty print) is used for better output interpretability
    pprint.pprint(t)


## OUTPUT
# {'outlook': {'Overcast': 'Yes',
#              'Rain': {'wind': {'Strong': 'No', 'Weak': 'Yes'}},
#              'Sunny': {'humidity': {'High': 'No', 'Normal': 'Yes'}}}}
