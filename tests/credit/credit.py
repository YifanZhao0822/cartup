from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('./crx.data.csv',header=None)

vars = [x for x in range(15)]
num_vars = [1, 2, 7, 10, 13, 14]
cat_vars = list(set(vars) - set(num_vars))
cat_vars[0] = '0'


# calculate the missing value rates
# missing_rate = defaultdict()
missing_rate = []
for column in vars:
    missing_rows = [x for x in range(data.shape[0]) if data.iloc[x][column] == '?']
    missing_count = len(missing_rows)
    # missing_rate[column] = missing_count / data.shape[0]
    missing_rate.append(missing_count / data.shape[0])
# missing_rate = pd.DataFrame(missing_rate)
# missing_rate.reset_index(inplace=True)
# missing_rate = missing_rate.rename(columns={'index': 'Column', 0: 'Cost'})
# missing_rate.to_csv('Missing_Value_Rate.csv', header=True)


for column in num_vars:
    missing_rows = [x for x in range(data.shape[0]) if data.iloc[x][column] == '?']
    for row in missing_rows:
        data.loc[row, column] = np.NaN
    nonmissing_rows = list(set(range(data.shape[0])) - set(missing_rows))

    data.loc[:, column] = data.loc[:, column].astype('float32')
    column_mean = data.loc[:, column].mean()
    for row in missing_rows:
        data.loc[row, column] = column_mean

data.to_csv('credit_data.csv')

# data_cat = data.drop([1, 2, 7, 10, 13, 14], axis=1)
# data = data.astype({0: str, 2: str, 4: str, 19: str})
# data_dummy = pd.get_dummies(data, prefix=cat_vars + [15], drop_first=True)
# X = data_dummy.loc[:, data_dummy.columns != '15_-']
# y = data_dummy.loc[:, data_dummy.columns == '15_-']

# for column in num_vars:
#     column_mean = X.loc[:, column].mean()
#     column_std = X.loc[:, column].std()
#     new_column = str(column) + '_below'
#     X[new_column] = (X.loc[:, column] - column_mean) < -column_std
#     X[new_column] = X[new_column].astype('int32')
#     new_column = str(column) + '_beyond'
#     X[new_column] = (X.loc[:, column] - column_mean) > column_std
#     X[new_column] = X[new_column].astype('int32')
# X = X.drop(columns=num_vars)

X = data.loc[:, data.columns != 15]
y = data.loc[:, data.columns == 15]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=123, shuffle=True)

val_positive = y_val == '-'
val_positive.rename(columns={15: 0}, inplace=True)
y_val_positive = y_val.loc[val_positive[0], :]
X_val_positive = X_val.loc[val_positive[0], :]
y_val_negative = y_val.loc[~val_positive[0], :]
X_val_negative = X_val.loc[~val_positive[0], :]

test_positive = y_test == '-'
test_positive.rename(columns={15: 0}, inplace=True)
y_test_positive = y_test.loc[test_positive[0], :]
X_test_positive = X_test.loc[test_positive[0], :]
y_test_negative = y_test.loc[~test_positive[0], :]
X_test_negative = X_test.loc[~test_positive[0], :]

# cost = pd.read_csv('./Missing_Value_Rate.csv', header=0)
# cost = pd.DataFrame({'cost': cost})
cost = pd.DataFrame({'cost': missing_rate})
cost.to_csv('credit_cost.csv')

# convert to list for input
cost_array = np.array(cost)
cost_array = cost_array.reshape([-1, 1])

rank = defaultdict(lambda: 0)
for feature in cost_array:
    if feature <= 0.00647:
        rank[0] += 1
    elif feature <= 0.2:
        rank[1] += 1
    elif feature <= 0.5:
        rank[2] += 1
    elif feature <= 0.9:
        rank[3] += 1
    else:
        rank[4] += 1

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
standardized_cost = min_max_scaler.fit_transform(cost_array).tolist()
standardized_cost = [x for subarray in standardized_cost for x in subarray]


from cartup import Tree, Data, DDist

DIS = 'discrete'
CLS = 'class'
NOM = 'nominal'

# formatting the data into expected input format
data_formatted = data.values.tolist()
X_train_formatted = X_train.values.tolist()
y_train_formatted = y_train.values.tolist()

data_train_formatted = X_train_formatted
for row in range(len(X_train_formatted)):
    data_train_formatted[row].append(y_train_formatted[row][0])


X_val_positive_formatted = X_val_positive.values.tolist()
y_val_positive_formatted = y_val_positive.values.tolist()

data_val_positive_formatted = X_val_positive_formatted
for row in range(len(X_val_positive_formatted)):
    data_val_positive_formatted[row].append(y_val_positive_formatted[row][0])

X_val_negative_formatted = X_val_negative.values.tolist()
y_val_negative_formatted = y_val_negative.values.tolist()

data_val_negative_formatted = X_val_negative_formatted
for row in range(len(X_val_negative_formatted)):
    data_val_negative_formatted[row].append(y_val_negative_formatted[row][0])


X_test_positive_formatted = X_test_positive.values.tolist()
y_test_positive_formatted = y_test_positive.values.tolist()

data_test_positive_formatted = X_test_positive_formatted
for row in range(len(X_test_positive_formatted)):
    data_test_positive_formatted[row].append(y_test_positive_formatted[row][0])

X_test_negative_formatted = X_test_negative.values.tolist()
y_test_negative_formatted = y_test_negative.values.tolist()

data_test_negative_formatted = X_test_negative_formatted
for row in range(len(X_test_negative_formatted)):
    data_test_negative_formatted[row].append(y_test_negative_formatted[row][0])

# order list of features
order = X_train.columns.values.tolist()
# order[0] = '0'
order.append(15)
order = [str(x) for x in order]

# types of features
types = {}
for column in order[:-1]:
    types[column] = NOM
# for column in num_vars:
#     types[column] = DIS

types[order[-1]] = NOM

modes = {order[-1]: CLS}

# costs of features
cost_dict = {}
cost_vars = data.columns.tolist()
cost_vars.remove(15)
cost_vars = [str(x) for x in cost_vars]
for i in range(len(cost_vars)):
    cost_dict[cost_vars[i]] = standardized_cost[i]

# create the fitting data object
'''notes: class variable index can not be 0
'''
data_train_fitting = Data(data_train_formatted, order=order, types=types, modes=modes)
data_val_positive_fitting = Data(data_val_positive_formatted, order=order, types=types, modes=modes)
data_val_negative_fitting = Data(data_val_negative_formatted, order=order, types=types, modes=modes)
data_test_positive_fitting = Data(data_test_positive_formatted, order=order, types=types, modes=modes)
data_test_negative_fitting = Data(data_test_negative_formatted, order=order, types=types, modes=modes)

# training and test
total_cost = sum(standardized_cost)
tree = Tree.build(data_train_fitting, coe=0.5, costs=cost_dict, max_cost=0.8*total_cost)
result_positive = tree.test(data_val_positive_fitting)
TP = result_positive.mean
result_negative = tree.test(data_val_negative_fitting)
TN = result_negative.mean

FN = 1 - TP
FP = 1 - TN
TP = TP * len(y_val_positive)
TN = TN * len(y_val_negative)
FN = FN * len(y_val_positive)
FP = FP * len(y_val_negative)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_cartup = 2 * precision * recall / (precision + recall)


# Comparison with OSDT
import sys
import os
sys.path.append(os.path.abspath("/Users/yifanzhao/Documents/Research/KEIKIDataHawaii/benchmark/OSDT/src"))
from osdt import bbound, predict


# data_osdt = data.copy()
# data_osdt = data_osdt.drop([1, 2, 7, 10, 13, 14], axis=1)
# data_osdt = pd.get_dummies(data_osdt, prefix=data_osdt.columns, drop_first=True)
# # data_osdt = data_osdt.drop('15_-', axis=1)
# # unique_columns = {'col1_0': ['col1_00', 'col1_01'], 'col1_1': ['col1_10', 'col1_11']}
# # data_osdt = data_osdt.rename(columns=lambda c: unique_columns[c].pop(0) if c in unique_columns.keys() else c)
#
# for column in data_osdt.columns:
#     if set(data_osdt[column].unique().tolist()) != set([0, 1]):
#         print("column %d not binary", column)
# print("test end")
#
# X_osdt = data_osdt.loc[:, data_osdt.columns != '15_-']
# y_osdt = data_osdt.loc[:, data_osdt.columns == '15_-']

X_osdt = X.copy()
y_osdt = y.copy()

X_train_osdt, X_test_osdt, y_train_osdt, y_test_osdt = train_test_split(X_osdt, y_osdt, test_size=0.3, random_state=123, shuffle=True)
X_train_osdt, X_val_osdt, y_train_osdt, y_val_osdt = train_test_split(X_train_osdt, y_train_osdt, test_size=0.15, random_state=123, shuffle=True)

X_train_osdt = X_train_osdt.values
y_train_osdt = y_train_osdt.values[:, 0]

val_positive = y_val_osdt == 0
val_positive.rename(columns={'15_-': 0}, inplace=True)
y_val_positive_osdt = y_val_osdt.loc[val_positive[0], :]
X_val_positive_osdt = X_val_osdt.loc[val_positive[0], :]
y_val_negative_osdt = y_val_osdt.loc[~val_positive[0], :]
X_val_negative_osdt = X_val_osdt.loc[~val_positive[0], :]

test_positive = y_test_osdt == 0
test_positive.rename(columns={'15_-': 0}, inplace=True)
y_test_positive_osdt = y_test_osdt.loc[test_positive[0], :]
X_test_positive_osdt = X_test_osdt.loc[test_positive[0], :]
y_test_negative_osdt = y_test_osdt.loc[~test_positive[0], :]
X_test_negative_osdt = X_test_osdt.loc[~test_positive[0], :]

X_val_positive_osdt = X_val_positive_osdt.values
y_val_positive_osdt = y_val_positive_osdt.values[:, 0]

X_val_negative_osdt = X_val_negative_osdt.values
y_val_negative_osdt = y_val_negative_osdt.values[:, 0]

X_test_positive_osdt = X_test_positive_osdt.values
y_test_positive_osdt = y_test_positive_osdt.values[:, 0]

X_test_negative_osdt = X_test_negative_osdt.values
y_test_negative_osdt = y_test_negative_osdt.values[:, 0]

lamb = 0.05
timelimit = 1800
tic = time.time()
leaves_c, prediction_c, dic, nleaves_OSDT, nrule, ndata, totaltime, time_c, COUNT, C_c, trainaccu_OSDT, best_is_cart, clf, children_pop, children_split, children_best = \
    bbound(X_train_osdt, y_train_osdt, lamb=lamb, prior_metric="curiosity", timelimit=timelimit, init_cart=True)
_, TP = predict(leaves_c, prediction_c, dic, X_val_positive_osdt, y_val_positive_osdt, best_is_cart, clf)
_, TN = predict(leaves_c, prediction_c, dic, X_val_negative_osdt, y_val_negative_osdt, best_is_cart, clf)

FN = 1 - TP
FP = 1 - TN
TP = TP * len(y_val_positive_osdt)
TN = TN * len(y_val_negative_osdt)
FN = FN * len(y_val_positive_osdt)
FP = FP * len(y_val_negative_osdt)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_osdt = 2 * precision * recall / (precision + recall)
time_osdt = time.time() - tic

# get the unique features
uniq_features = []
visited = set()
for tree in children_split:
    for leaf in tree.leaves:
        for feature in leaf.rules:
            feat = dic[abs(feature)]
            if feat not in visited:
                visited.add(feat)
                uniq_features.append(feat)


