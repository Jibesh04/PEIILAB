import pandas as pd
import numpy as np
import NBC1
eps = np.finfo(float).eps

def calc_prob(df):
    y = df.keys()[0]
    classes = df[y].unique()
    attributes = df.keys()[1:-1]
    df.set_index(y, inplace=True)

    N = 0
    for value in df['Total']:
        N = N + value
    pp = {}
    for cl in classes:
        pp[cl] = df['Total'].loc[cl]/N
    
    cp = {}
    for x in attributes:
        cp[x] = {}
        cp[x]['Yes'] = {}
        cp[x]['No'] = {}
        for cl in classes:
            num = df[x].loc[cl]
            den = df['Total'].loc[cl]
            cp[x]['Yes'][cl] = num/(den + eps)
            cp[x]['No'][cl] = 1 - cp[x]['Yes'][cl]
    
    return y, classes, cp, pp

def csv_to_list_of_dicts(df):
    data_list = []
    for idx, row in df.iterrows():
        data_list.append(row.to_dict())
    return data_list

def NaiveBayesClassifier(test_data):
    df2 = pd.DataFrame(test_data)
    list_of_classes = []
    list_of_prob = []
    for entry in csv_to_list_of_dicts(test_data):
        cl, pr = NBC1.classify(entry, classes, cp, pp)
        list_of_classes.append(cl)
        list_of_prob.append(pr)
    df2['Classification'] = list_of_classes
    df2['Probability'] = list_of_prob
    print(df2)

if __name__ == "__main__":
    df = pd.DataFrame(pd.read_csv('fruitdataset.csv'))
    y, classes, cp, pp = calc_prob(df)
    test_data = pd.read_csv('testfruit.csv')
    NaiveBayesClassifier(test_data)
    
# OUTPUT
#   Yellow Sweet Long Classification  Probability
# 0    Yes   Yes  Yes         Banana     0.218750
# 1    Yes   Yes   No          Mango     0.201923
# 2    Yes    No  Yes         Banana     0.072917
# 3    Yes    No   No          Mango     0.089744
# 4     No   Yes  Yes         Others     0.018519
# 5     No   Yes   No          Mango     0.173077
# 6     No    No  Yes         Others     0.009259
# 7     No    No   No          Mango     0.076923

# fruitdataset.csv
# Fruit,Yellow,Sweet,Long,Total
# Mango,350,450,0,650
# Banana,400,300,350,400
# Others,50,100,50,150
    
# testfruit.csv
# Yellow,Sweet,Long
# Yes,Yes,Yes
# Yes,Yes,No
# Yes,No,Yes
# Yes,No,No
# No,Yes,Yes
# No,Yes,No
# No,No,Yes
# No,No,No