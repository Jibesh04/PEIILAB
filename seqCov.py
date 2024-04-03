# Sequential Covering
def sequential_covering(examples):
    rules = []
    remaining_examples = examples.copy()

    while not remaining_examples.empty:
        rule = generate_rule(remaining_examples)
        if rule not in rules:
            rules.append(rule)
        else:
            break
        remaining_examples = remove_covered_examples(remaining_examples, rule)

    return rules


def generate_rule(examples):
    rule = {}
    for attribute in examples.columns[:]:
        value = examples[attribute].mode()[0]
        rule[attribute] = value

    return rule


def remove_covered_examples(examples, rule):
    for i, row in enumerate(examples):
        flag = True
        for attribute, value in rule.items():
            flag = flag and examples[attribute].loc[examples.index[i]] == value
        if flag:
            examples = examples.drop(examples.index[i])
    return examples

# Example usage
import pandas as pd

# Example dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

rules = sequential_covering(df)
for i, rule in enumerate(rules):
    print(f"Rule {i+1}: {rule}")