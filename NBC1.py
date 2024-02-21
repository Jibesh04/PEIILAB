import pandas as pd
import numpy as np
eps = np.finfo(float).eps

def classify(entry, classes, cp, pp):
    max_prob = 0
    prob = 0
    class_of_entry = classes[0]
    for cl in classes:
        prob = pp[cl]
        for attr in entry:
            val = entry[attr]
            prob = prob * cp[attr][val][cl]
        if prob > max_prob:
            max_prob = prob
            class_of_entry = cl
    return class_of_entry, max_prob

def calc_prob(df):
    y = df.keys()[-1]
    classes = df[y].unique()
    attributes = df.keys()[:-1]
    
    N = len(df)
    pp = {}
    for cl in classes:
        c_count = len(df[df[y] == cl])
        pp[cl] = c_count/N
    
    cp = {}
    for x in attributes:
        values = df[x].unique()
        cp[x] = {}
        for val in values:
            cp[x][val] = {}
            for cl in classes:
                num = len(df[x] [df[x] == val] [df[y] == cl])
                den = len(df[df[y] == cl])
                cp[x][val][cl] = num/(den + eps)
    
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
        cl, pr = classify(entry, classes, cp, pp)
        list_of_classes.append(cl)
        list_of_prob.append(pr)
    df2['Classification'] = list_of_classes
    df2['Probability'] = list_of_prob
    print(df2)

if __name__ == "__main__":
    df = pd.DataFrame(pd.read_csv('feverdataset.csv'))
    y, classes, cp, pp = calc_prob(df)
    # entry = {'COVID' : 'No', 'Flu' : 'No'}
    # predicted_class = classify(entry, classes, cp, pp)
    # print(f'{y}:{predicted_class}')
    test_data = pd.read_csv('testfever.csv')
    NaiveBayesClassifier(test_data)

# OUTPUT
#   COVID  Flu Classification  Probability
# 0   Yes  Yes            Yes     0.171429
# 1   Yes   No            Yes     0.228571
# 2    No  Yes            Yes     0.128571
# 3    No   No            Yes     0.171429
    
# feverdataset.csv
# COVID,Flu,Fever
# Yes,No,Yes
# No,Yes,Yes
# Yes,Yes,Yes
# No,No,No
# Yes,No,Yes
# No,No,Yes
# Yes,No,Yes
# Yes,No,No
# No,Yes,Yes
# No,Yes,No
    
# testfever.csv
# COVID,Flu
# Yes,Yes
# Yes,No
# No,Yes
# No,No