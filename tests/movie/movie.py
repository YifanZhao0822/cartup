from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_excel('./movie_dataset.xlsx',header=0)
data = data.loc[:, data.columns != 'Movie']

vars = data.columns.tolist()
nom_vars = ['Year', 'Genre', 'Sequel', 'Sentiment']
num_vars = list(set(vars) - set(nom_vars))

# calculate the missing value rates
# missing_rate = defaultdict()
missing_rate = []
for column in vars:
    missing_rows = [x for x in range(data.shape[0]) if pd.isnull(data.iloc[x][column])]
    missing_count = len(missing_rows)
    # missing_rate[column] = missing_count / data.shape[0]
    missing_rate.append(missing_count / data.shape[0])
# missing_rate = pd.DataFrame(missing_rate)
# missing_rate.reset_index(inplace=positive)
# missing_rate = missing_rate.rename(columns={'index': 'Column', 0: 'Cost'})
# missing_rate.to_csv('Missing_Value_Rate.csv', header=positive)


for column in num_vars:
    missing_rows = [x for x in range(data.shape[0]) if pd.isnull(data.iloc[x][column])]
    for row in missing_rows:
        data.loc[row, column] = np.NaN
    nonmissing_rows = list(set(range(data.shape[0])) - set(missing_rows))

    data.loc[:, column] = data.loc[:, column].astype('float32')
    column_mean = data.loc[:, column].mean()
    for row in missing_rows:
        data.loc[row, column] = column_mean

    data[column] /= (column_mean / 10)
    if column != 'Aggregate Followers':
        data[column] = data[column].astype(int)

data.to_csv("movie_data.csv")

X = data.loc[:, data.columns != 'Aggregate Followers']
y = data.loc[:, data.columns == 'Aggregate Followers']

# data_dummy = pd.get_dummies(X, prefix=vars[:-1], drop_first=True)
# X = data_dummy.loc[:, data_dummy.columns != 'Aggregate Followers']
# y = data_dummy.loc[:, data_dummy.columns == 'Aggregate Followers']
#
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=132, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=132, shuffle=True)

cost = pd.DataFrame({'cost': missing_rate})
cost.to_csv('movie_cost.csv')

# convert to list for input
cost_array = np.array(cost)
cost_array = cost_array.reshape([-1, 1])

rank = defaultdict(lambda: 0)
for feature in cost_array:
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

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
standardized_cost = min_max_scaler.fit_transform(cost_array).tolist()
standardized_cost = [x for subarray in standardized_cost for x in subarray]


from cartup import Tree, Data

DIS = 'discrete'
CLS = 'class'
NOM = 'nominal'
CON = 'continuous'

# formatting the data into expected input format
data_formatted = data.values.tolist()
X_train_formatted = X_train.values.tolist()
y_train_formatted = y_train.values.tolist()

data_train_formatted = X_train_formatted
for row in range(len(X_train_formatted)):
    data_train_formatted[row].append(y_train_formatted[row][0])


X_val_formatted = X_val.values.tolist()
y_val_formatted = y_val.values.tolist()

data_val_formatted = X_val_formatted
for row in range(len(X_val_formatted)):
    data_val_formatted[row].append(y_val_formatted[row][0])


X_test_formatted = X_test.values.tolist()
y_test_formatted = y_test.values.tolist()

data_test_formatted = X_test_formatted
for row in range(len(X_test_formatted)):
    data_test_formatted[row].append(y_test_formatted[row][0])


# order list of features
order = X_train.columns.values.tolist()
order.append('0')

# types of features
types = {}
for column in order[:-1]:
    types[column] = NOM
# for column in num_vars:
#     types[column] = DIS

types[order[-1]] = CON

modes = {order[-1]: CLS}

# costs of features
cost_dict = {}
cost_vars = data.columns.tolist()
cost_vars.remove('Aggregate Followers')
for i in range(len(cost_vars)):
    cost_dict[cost_vars[i]] = standardized_cost[i]

# create the fitting data object
'''notes: class variable index can not be 0
'''
data_train_fitting = Data(data_train_formatted, order=order, types=types, modes=modes)
data_val_fitting = Data(data_val_formatted, order=order, types=types, modes=modes)
data_test_fitting = Data(data_test_formatted, order=order, types=types, modes=modes)

# training and test
total_cost = sum(standardized_cost)
tree = Tree.build(data_train_fitting, coe=0.5, costs=cost_dict, max_cost=0.8*total_cost)
result = tree.test(data_val_fitting)
mae_cartup = result.mean