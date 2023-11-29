import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import graphviz

##################################  Data pre-processing ##################################

# reading the red wine data
df_red = pd.read_csv('winequality-red.csv', sep=';')
# replacing space with underscore in column headers
df_red.columns = df_red.columns.str.replace(' ', '_')

# how many rows and columns
print(df_red.shape)
df_red.head()

# adding a column for color = red
# df_red["color"] = pd.Series(["red" for i in range(df_red.shape[0])])
# df_red

# reading the white wine data
df_white = pd.read_csv('winequality-white.csv', sep=';')
# replacing space with underscore in column headers
df_white.columns = df_white.columns.str.replace(' ', '_')

# how many rows and columns
print(df_white.shape)
df_white.head()

# adding a column for color = white
# df_white["color"] = pd.Series(["white" for i in range(df_white.shape[0])])
# df_white

# combine csv files
df = pd.concat([df_red, df_white], ignore_index=True)
print('Totals for combined CSV: \n\n', df.shape)

# transforming color column to white = 1 and red = 0
# df["color"] = LabelEncoder().fit_transform(df["color"])
# df

##################################  Data cleaning ##################################

# Check for NULL values
print('***Number of NULL values per column***')
print(df.isna().sum(), end='\n\n')

# Check for duplicates
print('Totals before cleaning: ', df.shape)
print('Duplicates : ', df.duplicated().sum())

# Drop the duplicates
df.drop_duplicates(inplace=True)

# totals after duplicates dropped
print('Totals after cleaning: \n', df.shape)

##################################  EDA ##################################

warnings.filterwarnings('ignore')

# show histplot for each column
plt.figure(figsize=(15, 10))
plot_num = 1
for i in df.columns:
    ax = plt.subplot(4, 4, plot_num)
    sns.histplot(df[i], kde=True)
    plot_num += 1
plt.tight_layout()
plt.show()

# correlation map

df.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt=".2f", ax=ax)
plt.show()

# show multivariate analysis
sns.pairplot(df, hue='quality', height=3)
plt.tight_layout()
plt.show()

# show scatterplot analysis
fig, axes = plt.subplots(11, 11, figsize=(50, 50))
for i in range(11):
    for j in range(11):
        axes[i, j].scatter(df.iloc[:, i], df.iloc[:, j], c=df.quality)
        axes[i, j].set_xlabel(df.columns[i])
        axes[i, j].set_ylabel(df.columns[j])
        axes[i, j].legend(df.quality)
plt.tight_layout()
plt.show()

g = sns.pairplot(df, hue="quality")

# distribution of quality rankings
sns.barplot(x=df['quality'].unique(), y=df['quality'].value_counts())
plt.xlabel("Quality Rankings")
plt.ylabel("Number of Wine")
plt.title("Distribution of Wine Quality Ratings")
plt.show()


# Check the outliers for each feature with respect to output value
fig, ax1 = plt.subplots(4, 3, figsize=(22, 16))
k = 0
for i in range(4):
    for j in range(3):
        if k != 11:
            sns.barplot(y=df.iloc[:, k], x='quality',
                        data=df, ax=ax1[i][j])
            k += 1
plt.tight_layout()
plt.show()

# Check the outliers for each feature with respect to output value
fig, ax1 = plt.subplots(4, 3, figsize=(22, 16))
k = 0
for i in range(4):
    for j in range(3):
        if k != 11:
            sns.boxplot(x='quality', y=df.iloc[:, k], data=df, ax=ax1[i][j])
            k += 1
plt.tight_layout()
plt.show()


##################################  Models- Classification and Cross Validation ##################################


# it gives for each value the same value intervals means between 0-1
def normalization(X):
    mean = np.mean(X)
    std = np.std(X)
    X_t = (X - mean)/std
    return X_t

# Train and Test splitting of data


def train_test(X_t, y):
    x_train, x_test, y_train, y_test = train_test_split(
        X_t, y, test_size=0.3, random_state=42)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    print("Train:", len(x_train), " - Test:", len(x_test))
    return x_train, x_test, y_train, y_test


def grid_search(name_clf, clf, x_train, x_test, y_train, y_test):
    if name_clf == 'Logistic_Regression':
        # Logistic Regression
        log_reg_params = {"penalty": ['l1', 'l2'],
                          'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
        grid_log_reg.fit(x_train, y_train)
        # We automatically get the logistic regression with the best parameters.
        log_reg = grid_log_reg.best_estimator_
        print("Best Parameters for Logistic Regression: ",
              log_reg)
        print("Best Score for Logistic Regression: ", grid_log_reg.best_score_)
        print("------------------------------------------")

        return log_reg

    elif name_clf == 'Decision_Tree':
        # DecisionTree Classifier
        tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 30, 1)),
                       "min_samples_leaf": list(range(5, 20, 1))}
        grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
        grid_tree.fit(x_train, y_train)
        # tree best estimator
        tree_clf = grid_tree.best_estimator_
        print("Best Parameters for Decision Tree: ", grid_tree.best_estimator_)
        print("Best Score for Decision Tree: ", grid_tree.best_score_)
        print("------------------------------------------")

        # FEATURE IMPORTANCE FOR DECISION TREE
        importnce = tree_clf.feature_importances_
        plt.figure(figsize=(10, 10))
        plt.title("Feature Importances of Decision Tree")
        plt.barh(X_t.columns, importnce, align="center")

        return tree_clf

    elif name_clf == 'Random_Forest':
        forest_params = {"bootstrap": [True, False], "max_depth": list(range(2, 10, 1)),
                         "min_samples_leaf": list(range(5, 20, 1))}
        grid_forest = GridSearchCV(RandomForestClassifier(), forest_params)
        grid_forest.fit(x_train, y_train)
        # forest best estimator
        forest_clf = grid_forest.best_estimator_
        print("Best Parameters for Random Forest: ",
              grid_forest.best_estimator_)
        print("Best Score for Random Forest: ", grid_forest.best_score_)
        print("------------------------------------------")

        # FEATURE IMPORTANCE FOR DECISION TREE
        importnce = forest_clf.feature_importances_
        plt.figure(figsize=(10, 10))
        plt.title("Feature Importances of Random Forest")
        plt.barh(X_t.columns, importnce, align="center")
        return forest_clf


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv,
                                                            n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

# Create applying classification function


def apply_classification(name_clf, clf, x_train, x_test, y_train, y_test):
    # Find the best parameters and get the classification with the best parameters as return value of grid search
    grid_clf = grid_search(name_clf, clf, x_train, x_test, y_train, y_test)

    # Plotting the learning curve
    # score curves, each time with 30% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    plot_learning_curve(grid_clf, name_clf, x_train, y_train,
                        ylim=(0.1, 1.01), cv=cv, n_jobs=4)

    # Apply cross validation to estimate the skills of models with 10 split with using best parameters
    scores = cross_val_score(grid_clf, x_train, y_train, cv=10)
    print("Mean Accuracy of Cross Validation: %", round(scores.mean()*100, 2))
    print("Std of Accuracy of Cross Validation: %", round(scores.std()*100))
    print("------------------------------------------")

    # Predict the test data as selected classifier
    clf_prediction = grid_clf.predict(x_test)
    clf1_accuracy = sum(y_test == clf_prediction)/len(y_test)
    print("Accuracy of", name_clf, ":", clf1_accuracy*100)

    # print ROC curve for each model
    y_score_rfc = grid_clf.predict_proba(x_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rfc, pos_label=1)
    roc_auc_forests = auc(fpr_rf, tpr_rf)
    print(f"ROC Curve of {name_clf}:", roc_auc_forests)

    # print confusion matrix and accuracy score before best parameters
    clf1_conf_matrix = confusion_matrix(y_test, clf_prediction)
    print("Confusion matrix of", name_clf, ":\n", clf1_conf_matrix)
    print("==========================================")
    return grid_clf


# Now seperate the dataset as response variable and feature variabes
X = df.drop(['quality'], axis=1)
# y = pd.DataFrame(data['value'])
y = df['quality']
# Normalization
X_t = normalization(X)
print("X_t:", X_t.shape)

# Train and Test splitting of data
x_train, x_test, y_train, y_test = train_test(X_t, y)


lr = LogisticRegression()  # verbose=True
lr.fit(x_train, y_train)
y_pred_log = lr.predict(x_test)
lr_precision, lr_recall, _ = precision_recall_curve(
    y_test, y_pred_log, pos_label=1)
print('running LR classification')
apply_classification('Logistic_Regression', lr,
                     x_train, x_test, y_train, y_test)

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred_log = dt.predict(x_test)
dt_precision, dt_recall, _ = precision_recall_curve(
    y_test, y_pred_log, pos_label=1)
print('running DT classification')
dt_clf = apply_classification(
    'Decision_Tree', dt, x_train, x_test, y_train, y_test)

# Plot the decision tree
dot_data = export_graphviz(dt_clf, out_file=None,
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph

rf = RandomForestClassifier(n_estimators=100)  # , verbose=True
rf.fit(x_train, y_train)
y_pred_log = rf.predict(x_test)
rf_precision, rf_recall, _ = precision_recall_curve(
    y_test, y_pred_log, pos_label=1)
print('running RF classification')
apply_classification('Random_Forest', rf, x_train, x_test, y_train, y_test)

# # Precision recall curves analysis

no_skill = len(y_test[y_test == 1]) / len(y_test)

t = np.arange(0, 110, 10)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dt_recall, dt_precision, linestyle='--',
        color='b',  lw=2.0, label='DT')
ax.plot(lr_recall, lr_precision, linestyle='--',
        color='y',  lw=2.0, label='LR')
ax.plot(rf_recall, rf_precision, linestyle='--',
        color='m',  lw=2.0, label='RF')

ax.plot([0, 1], [no_skill, no_skill], marker='.',
        color='g', lw=2.0, label='No skill')

plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision-Recall curve', fontsize=18)
plt.legend(loc='best', fontsize=18)
ax.grid(True)
ticklines = ax.get_xticklines() + ax.get_yticklines()
gridlines = ax.get_xgridlines()
ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

for line in ticklines:
    line.set_linewidth(3)

for line in gridlines:
    line.set_linestyle('-')

for line in gridlines:
    line.set_linestyle('-')

for label in ticklabels:
    label.set_color('black')
    label.set_fontsize('large')

plt.show()

##### Binary dataset analysis ##############################################

# Add a new feature according to mean of the quality
# Good wine represented by 1, bad wine represented by 0
df['value'] = ""
df['value'] = [1 if each > 5 else 0 for each in df['quality']]

print("Good Wine Class:", df[df['value'] == 1].shape)
print("Bad Wine Class:", df[df['value'] == 0].shape)

# Check the outliers for each feature with respect to output value
fig, ax1 = plt.subplots(4, 3, figsize=(22, 16))
k = 0
for i in range(4):
    for j in range(3):
        if k != 11:
            sns.boxplot(x='value', y=df.iloc[:, k], data=df, ax=ax1[i][j])
            k += 1
plt.tight_layout()
plt.show()

# Categorical distribution plots:
fig, ax1 = plt.subplots(4, 3, figsize=(22, 16))
k = 0
for i in range(4):
    for j in range(3):
        if k != 11:
            sns.barplot(x="value", y=df.iloc[:, k],
                        hue='value', data=df, ax=ax1[i][j])
            k += 1
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(11, 11, figsize=(50, 50))
for i in range(11):
    for j in range(11):
        axes[i, j].scatter(df.iloc[:, i], df.iloc[:, j], c=df.value)
        axes[i, j].set_xlabel(df.columns[i])
        axes[i, j].set_ylabel(df.columns[j])
        axes[i, j].legend(df.value)
        leg = plt.legend(loc='upper center')
# plt.tight_lay out()
plt.show()

# Now seperate the dataset as response variable and feature variabes
Xb = df.drop(['quality', 'value'], axis=1)
# y = pd.DataFrame(data['value'])
yb = df['value']

# Normalization
Xb_t = normalization(Xb)
print("X_t:", Xb_t.shape)

# Train and Test splitting of data
xb_train, xb_test, yb_train, yb_test = train_test(Xb_t, yb)

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred_log = lr.predict(x_test)
lr_precision, lr_recall, _ = precision_recall_curve(
    y_test, y_pred_log, pos_label=1)
print('running LR binary classification')
apply_classification('Logistic_Regression', lr,
                     xb_train, xb_test, yb_train, yb_test)

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred_log = dt.predict(x_test)
dt_precision, dt_recall, _ = precision_recall_curve(
    y_test, y_pred_log, pos_label=1)
print('running DT binary classification')
dtb_clf = apply_classification(
    'Decision_Tree', dt, xb_train, xb_test, yb_train, yb_test)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
y_pred_log = rf.predict(x_test)
rf_precision, rf_recall, _ = precision_recall_curve(
    y_test, y_pred_log, pos_label=1)
print('running RF binary classification')
apply_classification('Random_Forest', rf, xb_train,
                     xb_test, yb_train, yb_test)

# Plot the decision tree
dot_data = export_graphviz(dtb_clf, out_file=None,
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph

# Precision recall curves analysis

no_skill = len(y_test[y_test == 1]) / len(y_test)

t = np.arange(0, 110, 10)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dt_recall, dt_precision, linestyle='--',
        color='b',  lw=2.0, label='DT')
ax.plot(lr_recall, lr_precision, linestyle='--',
        color='y',  lw=2.0, label='LR')
ax.plot(rf_recall, rf_precision, linestyle='--',
        color='m',  lw=2.0, label='RF')

ax.plot([0, 1], [no_skill, no_skill], marker='.',
        color='g', lw=2.0, label='No skill')

plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision-Recall curve', fontsize=18)
plt.legend(loc='best', fontsize=18)
ax.grid(True)
ticklines = ax.get_xticklines() + ax.get_yticklines()
gridlines = ax.get_xgridlines()
ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

for line in ticklines:
    line.set_linewidth(3)

for line in gridlines:
    line.set_linestyle('-')

for line in gridlines:
    line.set_linestyle('-')

for label in ticklabels:
    label.set_color('black')
    label.set_fontsize('large')

plt.show()
