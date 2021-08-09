from collections import defaultdict
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('./ad.data.csv',header=None)
data.rename(columns={0: '0'}, inplace=True)
vars = [x for x in range(1558)]
vars[0] = str(vars[0])
num_vars = ['0', 1, 2]
cat_vars = list(set(vars) - set(num_vars))

regex = re.compile('.*\?.*')

# calculate the missing value rates
# # missing_rate = defaultdict()
# missing_rate = []
# for column in vars:
#     missing_rows = [x for x in range(data.shape[0]) if regex.search(str(data.iloc[x][column]))]
#     missing_count = len(missing_rows)
#     # missing_rate[column] = missing_count / data.shape[0]
#     missing_rate.append(missing_count / data.shape[0])
# missing_rate_df = pd.DataFrame(missing_rate)
# # missing_rate.reset_index(inplace=True)
# # missing_rate = missing_rate.rename(columns={'index': 'Column', 0: 'Cost'})
# missing_rate_df.to_csv('./Missing_Value_Rate.csv', header=True)

for column in num_vars:
    missing_rows = [x for x in range(data.shape[0]) if regex.search(data.iloc[x][column])]
    for row in missing_rows:
        data.loc[row, column] = np.NaN
    nonmissing_rows = list(set(range(data.shape[0])) - set(missing_rows))

    data.loc[:, column] = data.loc[:, column].astype('float32')
    column_mean = data.loc[:, column].mean()
    for row in missing_rows:
        data.loc[row, column] = column_mean

data.to_csv('ad_data.csv')
# data_osdt = pd.get_dummies(data_osdt, prefix=['col1', 'col2'], drop_first=True)
# # data_osdt = data_osdt.drop('col2_nonad.', axis=1)
# unique_columns = {'col1_0': ['col1_00', 'col1_01'], 'col1_1': ['col1_10', 'col1_11']}
# data_dummy = pd.get_dummies(data, prefix=[3, 1558], drop_first=True)
# X = data_dummy.loc[:, data_dummy.columns != '1558_nonad.']
# y = data_dummy.loc[:, data_dummy.columns == '1558_nonad.']


# for column in num_vars:
#     column_mean = X.loc[:, column].mean()
#     column_std = X.loc[:, column].std()
#     new_column = str(column) + '_below'
#     X[new_column] = (X.loc[:, column] - column_mean) < -3*column_std
#     X[new_column] = X[new_column].astype('int32')
#     new_column = str(column) + '_beyond'
#     X[new_column] = (X.loc[:, column] - column_mean) > 3*column_std
#     X[new_column] = X[new_column].astype('int32')
# X = X.drop(columns=num_vars)

X = data.loc[:, data.columns != 1558]
y = data.loc[:, data.columns == 1558]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=123, shuffle=True)

val_positive = y_val == 'ad.'
val_positive.rename(columns={1558: 0}, inplace=True)
y_val_positive = y_val.loc[val_positive[0], :]
X_val_positive = X_val.loc[val_positive[0], :]
y_val_negative = y_val.loc[~val_positive[0], :]
X_val_negative = X_val.loc[~val_positive[0], :]

test_positive = y_test == 'ad.'
test_positive.rename(columns={1558: 0}, inplace=True)
y_test_positive = y_test.loc[test_positive[0], :]
X_test_positive = X_test.loc[test_positive[0], :]
y_test_negative = y_test.loc[~test_positive[0], :]
X_test_negative = X_test.loc[~test_positive[0], :]


# cost = [1.00] * X.shape[1]
cost = pd.read_csv('./Missing_Value_Rate.csv', index_col=0, header=0)
# cost = pd.DataFrame({'cost': cost})

rank = defaultdict(lambda: 0)
for feature in cost['0']:
    if feature <= 0.1:
        rank[0] += 1
    elif feature <= 0.2:
        rank[1] += 1
    elif feature <= 0.5:
        rank[2] += 1
    elif feature <= 0.9:
        rank[3] += 1
    else:
        rank[4] += 1

# convert to list for input
cost_array = np.array(cost)
cost_array = cost_array.reshape([-1, 1])

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
order.append(1558)
order = [str(x) for x in order]

# types of features
types = {}
for column in order[:-1]:
    types[column] = NOM

types[order[-1]] = NOM

modes = {order[-1]: CLS}

# costs of features
cost_dict = {}
cost_vars = data.columns.tolist()
cost_vars.remove(1558)
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
f1_csdt = 2 * precision * recall / (precision + recall)