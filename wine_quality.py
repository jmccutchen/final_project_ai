import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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
print('Totals for combined CSV: ', df.shape)

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
# -1 means: There is a negative relationship between dependent and independent variables .
# 0 means: There is no relationship between dependent and independent variables .
# 1 means: There is a positive relationship between dependent and independent variables .
# According to this information, it can be made a good analysis about dataset and columns.
df.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt=".2f", ax=ax)
plt.show()

# show multivariate analysis
# sns.pairplot(df, hue='quality', height=3)
# plt.tight_layout()
# plt.show()

# show scatterplot analysis
fig, axes = plt.subplots(11, 11, figsize=(25, 25))
for i in range(11):
    for j in range(11):
        axes[i, j].scatter(df.iloc[:, i], df.iloc[:, j], c=df.quality)
        axes[i, j].set_xlabel(df.columns[i])
        axes[i, j].set_ylabel(df.columns[j])
        axes[i, j].legend(df.quality)
plt.tight_layout()
plt.show()

# distribution of quality rankings
sns.barplot(x=df['quality'].unique(), y=df['quality'].value_counts())
plt.xlabel("Quality Rankings")
plt.ylabel("Number of Red Wine")
plt.title("Distribution of Red Wine Quality Ratings")
plt.show()

# Check the outliers for each feature with respect to output value
fig, ax1 = plt.subplots(4, 3, figsize=(22, 16))
k = 0
for i in range(4):
    for j in range(3):
        if k != 11:
            # need to figure out issue with this below, example is incorrect step 9 here https://www.kaggle.com/code/sevilcoskun/red-wine-quality-classification
            sns.barplot(hue='quality',
                        gap=df.iloc[:, k], data=df, ax=ax1[i][j])
            k += 1
plt.show()

# Check the outliers for each feature with respect to output value
fig, ax1 = plt.subplots(4, 3, figsize=(22, 16))
k = 0
for i in range(4):
    for j in range(3):
        if k != 11:
            sns.barplot('quality', df.iloc[:, k], data=df, ax=ax1[i][j])
            k += 1
plt.show()

##################################  Models ##################################

# Predictors and target
x = df.drop('quality', axis=1)
y = df.quality
x_train, x_test, y_train, y_test = train_test_split(x, y)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
confusion_matrix_lr = confusion_matrix(y_test, pred)
print('\n\nConfusion Matrix LR: \n\n', confusion_matrix_lr)
print('\n\nLogistic Regression Classifier: \n\n',
      classification_report(y_test, pred))


# Random Forest
xgb = RandomForestClassifier()
xgb.fit(x_train, y_train)
pred = xgb.predict(x_test)
print('Random Forest Classifier: \n\n', classification_report(y_test, pred))

# Linear Regression
regr = LinearRegression().fit(x_train, y_train)
y_hat = regr.predict(x_test)
y_hat[y_hat > 10] = 10
y_hat[y_hat < 0] = 0
y_hat = y_hat.round().astype("int")
print("Linear Regression Accuracy Score: {accur:.2f}".format(
    accur=accuracy_score(y_test, y_hat)))

# Decision Tree
dt = DecisionTreeClassifier(
    criterion="entropy", random_state=1).fit(x_train, y_train)
y_hat = dt.predict(x_test)
print("Decision Tree Accuracy Score: {accur:.2f}".format(
    accur=accuracy_score(y_test, y_hat)))

# Stochastic Gradient Descent Classifier

# Support Vector Classifier(SVC)
