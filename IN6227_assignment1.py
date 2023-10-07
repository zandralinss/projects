import pandas as pd
import math
import numpy as np
import time

pd.options.mode.chained_assignment = None


class DecisionTreeNode:
    def __init__(self, current_feature=None, current_binary_sets=None, left=None, right=None, value=None):
        #  building tree with node splitting feature, node splitting binary sets
        #  left, right, and node value.
        self.current_feature = current_feature
        self.current_binary_sets = current_binary_sets
        self.left = left
        self.right = right
        self.value = value


def cal_gini(y):
    gini = 1.0
    #  calculation based on panda series, so need to convert to panda series if input is a list.
    if isinstance(y, list):
        y = pd.Series(y)
    total_num = len(y)
    for value in y.value_counts():
        gini += -(value / total_num) ** 2
    return gini


def find_best_split(X, y):
    row = len(X)
    if row <= 1:
        return None, None
    y_values_list = y.value_counts()
    best_split_feature = None
    # get current split value
    best_split_gini = cal_gini(y)
    best_binary_subsets = []
    for feature_name in X.columns:
        gini_split = cal_gini(y)
        feature_values = X[feature_name]
        # to check if the type of feature is numerical or categorical
        is_numeric = pd.api.types.is_numeric_dtype(feature_values)
        if is_numeric:
            #  sorted the values first, easier to do splitting
            sorted_indices = np.argsort(feature_values)
            y_sorted = y.iloc[sorted_indices]
            #  get the list of values of features
            feature_sorted = list(feature_values.iloc[sorted_indices])
            #  all the value in the right, and would move to left gradually
            sub_left = [0, 0]
            sub_right = y_values_list.copy()
            subsets_left = []
            subsets_right = list(set(feature_sorted)).copy()
            binary_subsets = []
            for j in range(0, row - 1):
                current_value = y_sorted[j]
                sub_left[current_value] += 1
                sub_right[current_value] -= 1
                #  if next value same as current one, then pass it
                if feature_sorted[j] == feature_sorted[j + 1]:
                    continue
                gini_left = cal_gini([0] * sub_left[0] + [1] * sub_left[1])
                gini_right = cal_gini(sub_right)
                gini_split_cur = gini_left * ((j + 1) / row) + gini_right * ((row - j - 1) / row)
                #  split values into to subsets based on the value itself.
                #  if equal or less would be on the left
                #  others would be on the right
                subsets_left.append(feature_sorted[j])
                subsets_right.remove(feature_sorted[j])
                if gini_split > gini_split_cur:
                    gini_split = gini_split_cur
                    binary_subsets = [subsets_left.copy(), subsets_right.copy()]
        else:
            #  categorical features
            gini_split, binary_subsets = find_categorical_split(feature_values, y)

        #  get the best feature to split
        if gini_split < best_split_gini:
            best_split_gini = gini_split
            best_split_feature = feature_name
            best_binary_subsets = binary_subsets

    return best_split_feature, best_split_gini, best_binary_subsets


def find_categorical_split(x, y):
    x_set = set(x)
    #  to get all possible split binary sets for the feature
    possible_subsets = find_all_possible_subsets(sorted(list(x_set)))
    best_gini = cal_gini(y)
    best_subsets = []
    possible_subsets.sort(key=lambda subsets: subsets[0])
    # for loop to get the gini split value for all subsets
    for subsets in possible_subsets:
        left_leave = [0, 0]
        right_leave = [0, 0]
        left_indices = x[x.isin(subsets[0])]
        left_y = y[left_indices.index]
        left_count = len(left_y)
        # get 0, 1 values in the first subset of current binary split
        for y_value in left_y:
            left_leave[y_value] += 1
        right_indices = x[x.isin(subsets[1])]
        right_y = y[right_indices.index]
        right_count = len(right_y)
        # get 0, 1 values in the second subset of current binary split
        for y_value in right_y:
            right_leave[y_value] += 1
        gini_left = cal_gini([0] * left_leave[0] + [1] * left_leave[1])
        gini_right = cal_gini([0] * right_leave[0] + [1] * right_leave[1])
        gini_split = gini_left * (left_count / len(y)) + gini_right * (right_count / len(y))
        if best_gini > gini_split:
            best_gini = gini_split
            best_subsets = subsets.copy()
    return best_gini, best_subsets


def find_all_possible_subsets(value_list):
    result = []
    #  find out the possible binary splitting sets.
    for i in range(len(value_list)):
        sub_list1 = []
        sub_list2 = value_list.copy()
        for j in range(len(value_list)):
            if j >= i:
                sub_list1.append(value_list[j])
                sub_list2.remove(value_list[j])
                if [sub_list2.copy(), sub_list1.copy()] not in result:
                    if len(sub_list1.copy()) > 0 and len(sub_list2.copy()) > 0:
                        result.append([sub_list1.copy(), sub_list2.copy()])
    return result


def build_tree(X, y, depth=0, max_depth=None, min_samples_split=50, min_impurity_split=1e-7):
    #  get 0, 1 values count
    num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
    #  maximum one is the current class if the node would be a leaf
    predicted_class = np.argmax(num_samples_per_class)
    node = DecisionTreeNode(value=predicted_class)
    #  check the depth and current sample size, once reach the maximum depth or the minimum split size,
    #  the tree will not grow any more
    if depth >= max_depth or len(y) < min_samples_split:
        node.current_feature = None
        node.current_binary_sets = None
        return node

    #  if the tree is still small, then continue finding the splits to grow
    split_feature, split_gini, binary_subsets = find_best_split(X, y)
    if split_feature is None or split_gini < min_impurity_split:
        node.current_feature = None
        node.current_binary_sets = None
        return node

    #  split the current data into two subsets based on the binary subsets of the split feature
    indices_left = X.index[X[split_feature].isin(binary_subsets[0])].tolist()
    X_left, y_left = X.loc[indices_left], y.loc[indices_left]
    X_right, y_right = X.loc[~X.index.isin(indices_left)], y.loc[~X.index.isin(indices_left)]
    X_left.reset_index(drop=True, inplace=True)
    y_left.reset_index(drop=True, inplace=True)
    X_right.reset_index(drop=True, inplace=True)
    y_right.reset_index(drop=True, inplace=True)
    node.current_feature = split_feature
    node.current_binary_sets = binary_subsets
    #  iterate to build tree
    node.left = build_tree(X_left, y_left, depth + 1, max_depth)
    node.right = build_tree(X_right, y_right, depth + 1, max_depth)

    return node


def predict_tree(node, X):
    #  iterate to each a leaf to get the predicted value
    if node.left is None and node.right is None:
        return node.value
    if row[node.current_feature] in node.current_binary_sets[0]:
        return predict_tree(node.left, row)
    else:
        return predict_tree(node.right, X)


def data_preprocessing(data, data_test):
    #  drop the duplicates
    data = data.drop_duplicates(keep="first")
    #  re-code the target to be 0, 1 binary values
    data.replace('<=50K', 0, inplace=True)
    data.replace('>50K', 1, inplace=True)
    #  do same things for the test dataset
    data_test = data_test.drop_duplicates(keep="first")
    data_test.replace('<=50K.', 0, inplace=True)
    data_test.replace('>50K.', 1, inplace=True)

    #  replace ? with nan
    data.replace("?", np.nan, inplace=True)
    data_test.replace("?", np.nan, inplace=True)

    #  fill in the nan age value with the mode
    data_test['age'] = pd.to_numeric(data_test['age'], errors='coerce')
    data_test['age'].fillna(data["age"].mode()[0], inplace=True)
    #  converting the values into categories
    data['age'] = pd.cut(data['age'], bins=[0, 18, 40, 65, 100], labels=['juvenlies', 'early-adult', 'middle-age', 'the aged']).copy()
    data_test['age'] = pd.cut(data_test['age'], bins=[0, 18, 40, 65, 100], labels=['juvenlies', 'early-adult', 'middle-age', 'the aged']).copy()

    #  fill in the nan workclass value with the mode
    data["workclass"] = data["workclass"].fillna(data["workclass"].mode()[0])
    data_test["workclass"] = data_test["workclass"].fillna(data["workclass"].mode()[0])

    #  drop fnlwgt column
    data.drop(['fnlwgt'], axis=1, inplace=True)
    data_test.drop(['fnlwgt'], axis=1, inplace=True)

    #  drop education column
    data.drop(['education'], axis=1, inplace=True)
    data_test.drop(['education'], axis=1, inplace=True)

    #  fill in the nan education-num, marital-status, occupation, relationship, race, sex value with the mode
    data_test["education-num"] = data_test["education-num"].fillna(data["education-num"].mode()[0])

    data_test["marital-status"] = data_test["marital-status"].fillna(data["marital-status"].mode()[0])

    data["occupation"] = data["occupation"].fillna(data["occupation"].mode()[0])
    data_test["occupation"] = data_test["occupation"].fillna(data["occupation"].mode()[0])

    data_test["relationship"] = data_test["relationship"].fillna(data["relationship"].mode()[0])

    data_test["race"] = data_test["race"].fillna(data["race"].mode()[0])

    data_test["sex"] = data_test["sex"].fillna(data["sex"].mode()[0])

    #  calculate net capital gain
    data['net-capital-gain'] = data['capital-gain'] - data['capital-loss']

    data_test["capital-gain"] = data_test["capital-gain"].fillna(data["capital-gain"].mode()[0])
    data_test["capital-loss"] = data_test["capital-loss"].fillna(data["capital-loss"].mode()[0])
    data_test['net-capital-gain'] = data_test['capital-gain'] - data_test['capital-loss']

    #  drop capital-gain, capital-loss columns
    data = data.drop(['capital-gain'], axis=1).copy()
    data = data.drop(['capital-loss'], axis=1).copy()
    data_test = data_test.drop(['capital-gain'], axis=1).copy()
    data_test = data_test.drop(['capital-loss'], axis=1).copy()

    #  converting the values into categories
    data_test["hours-per-week"] = data_test["hours-per-week"].fillna(data["hours-per-week"].mode()[0])
    data['hours-per-week'] = pd.cut(data['hours-per-week'], bins=[0, 29, 49, 100], labels=['less hours', 'normal hours', 'over hours'])
    data_test['hours-per-week'] = pd.cut(data_test['hours-per-week'], bins=[0, 29, 49, 100], labels=['less hours', 'normal hours', 'over hours'])

    #  fill in the nan native-country value with the mode
    data['native-country'] = data['native-country'].fillna(data["native-country"].mode()[0])
    data.loc[data['native-country'] != 'United-States', 'native-country'] = 'Others'
    data_test['native-country'] = data_test['native-country'].fillna(data["native-country"].mode()[0])
    data_test.loc[data_test['native-country'] != 'United-States', 'native-country'] = 'Others'

    data.reset_index(drop=True, inplace=True)
    X = data.drop(['income'], axis=1)
    y = data['income']

    data_test = data_test[data_test['income'].notna()]
    data_test.reset_index(drop=True, inplace=True)
    X_test = data_test.drop(['income'], axis=1)
    y_test = data_test['income']

    return X, y, X_test, y_test


column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]  # Define column names based on the dataset description
data_train = pd.read_csv("/Users/shuangshuanglin/Downloads/NTU/IN6227/assignment1/Census Income Data Set/adult.data", names=column_names, sep=',\s*', engine='python')
data_test = pd.read_csv("/Users/shuangshuanglin/Downloads/NTU/IN6227/assignment1/Census Income Data Set/adult.test", names=column_names, sep=',\s*', engine='python')
X, y, X_test, y_test = data_preprocessing(data_train, data_test)

start_time = time.time()

tree_root = build_tree(X, y, max_depth=15)

#  record the times to build the tree
end_time = time.time()
execution_time = end_time - start_time
print(f"build tree time：{execution_time} seconds")

#  use to record the values for the performance matrics
matches = 0
tp = 0
tn = 0
fp = 0
fn = 0
total = len(y_test)
for index, row in X_test.iterrows():
    prediction = predict_tree(tree_root, X_test)
    if prediction == y_test[index]:
        matches += 1
        if prediction == 1:
            tp += 1
        else:
            tn += 1
    else:
        if prediction == 1:
            fp += 1
        else:
            fn += 1

accuracy = 100 * matches * 1.0 / total
print("Accuracy is: " + str(accuracy) + "%")
precision = 100 * tp * 1.0 / (tp + fp)
print("Precision is: " + str(precision) + "%")
recall = 100 * tp * 1.0 / (tp + fn)
print("Recall is: " + str(recall) + "%")
f_measure = 100 * tp * 2 * 1.0 / (2 * tp + fn + fp)
print("F-measure is: " + str(f_measure) + "%")
confusion_matrix_str = f"                       Predicted\n" \
                       f"                  Positive   Negative\n" \
                       f"Actual Positive   TP={tp}    FN={fn}\n" \
                       f"       Negative   FP={fp}    TN={tn}\n"

print("Confusion Matrix:")
print(confusion_matrix_str)
