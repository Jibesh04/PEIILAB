# Decision Tree
import pandas as pd
import numpy as np
import pprint
eps = np.finfo(float).eps

def find_entropy(df):
    target = df.keys()[-1]
    values = df[target].unique()
    entropy = 0
    for value in values:
        fraction = df[target].value_counts()[value] / len(df[target])
        entropy += -fraction * np.log2(fraction)
    return entropy

def find_entropy_attribute(df, attribute):
    target = df.keys()[-1]
    target_values = df[target].unique()
    values = df[attribute].unique()
    entropy_attribute = 0
    for value in values:
        entropy_each_feature = 0
        for target_value in target_values:
            num = len(df[attribute] [df[attribute] == value] [df[target] == target_value])
            den = len(df[attribute] [df[attribute] == value])
            fraction = num / (den + eps)
            entropy_each_feature += -fraction * np.log2(fraction + eps)
        fraction2 = den/len(df)
        entropy_attribute += -fraction2 * entropy_each_feature

    return abs(entropy_attribute)

def find_decider(df):
    info_gain = []
    for key in df.keys()[1:-1]:
        info_gain.append(find_entropy(df) - find_entropy_attribute(df, key))
    return df.keys()[1:-1][np.argmax(info_gain)]

def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


def buildTree(df, tree=None):
    target = df.keys()[-1]
    node = find_decider(df)
    values = np.unique(df[node])

    if tree is None:
        tree = {}
        tree[node] = {}

    for value in values:
        subtable = get_subtable(df, node, value)
        target_value, counts = np.unique(subtable[target], return_counts=True)

        if len(counts) == 1:
            tree[node][value] = target_value[0]
        else:
            tree[node][value] = buildTree(subtable)
    return tree

if __name__ == "__main__":
    df = pd.read_csv('tennisDT.csv')
    t = buildTree(df)
    pprint.pprint(t)

def classify_sample(sample, tree):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    if attribute in sample:
        value = sample[attribute]
        if value in tree[attribute]:
            return classify_sample(sample, tree[attribute][value])
    return "Unknown"

if __name__ == "__main__":
    df_pred = pd.read_csv('tennisPred.csv')
    df_pred['classification'] = df_pred.apply(lambda row: classify_sample(row.to_dict(), t), axis=1)
    print(df_pred[['outlook', 'temperature', 'humidity', 'wind', 'play', 'classification']])

## OUTPUT
# {'outlook': {'Overcast': 'Yes',
#              'Rain': {'wind': {'Strong': 'No', 'Weak': 'Yes'}},
#              'Sunny': {'humidity': {'High': 'No', 'Normal': 'Yes'}}}}
