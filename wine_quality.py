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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

##################################  Data pre-processing ##################################

# reading the red wine data
df_red = pd.read_csv('winequality-red.csv', sep=';')
# how many rows and columns
print(df_red.shape)
df_red.head()

#adding a column for color = red
df_red["color"] = pd.Series(["red" for i in range(df_red.shape[0])])
df_red

# reading the white wine data
df_white = pd.read_csv('winequality-white.csv', sep=';')

# how many rows and columns
print(df_white.shape)
df_white.head()

#adding a column for color = white
df_white["color"] = pd.Series(["white" for i in range(df_white.shape[0])])
df_white

#combine csv files
df = pd.concat([df_red, df_white], ignore_index=True)
print('Totals for combined CSV: ', df.shape)

# transforming color column to white = 1 and red = 0
df["color"] = LabelEncoder().fit_transform(df["color"])
df

##################################  Data cleaning ##################################

# Check for NULL values
print('***Number of NULL values per column***')
print(df.isna().sum(), end='\n\n')

# Check for duplicates
print('Totals before cleaning: ', df.shape)
print('Duplicates : ',df.duplicated().sum())

# Drop the duplicates
df.drop_duplicates(inplace=True)

# totals after duplicates dropped
print('Totals after cleaning: \n', df.shape)

##################################  EDA ##################################

warnings.filterwarnings('ignore')

# show histplot for each column
# plt.figure(figsize=(20,10))
# plot_num =1
# for i in df.columns:
#     ax = plt.subplot(4,4,plot_num)
#     sns.histplot(df[i], kde=True)
#     plot_num +=1
# plt.tight_layout()
# plt.show()

# show multivariate analysis
# sns.pairplot(df,hue='quality', height=3)
# plt.tight_layout()
# plt.show()

# show count of quality
# plt.figure(figsize=(6,3))
# sns.countplot(data=df,x='quality')
# plt.tight_layout()
# plt.show()

##################################  Models ##################################

# Predictors and target
x = df.drop('quality', axis=1)
y = df.quality
x_train, x_test, y_train, y_test = train_test_split(x,y)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
print('Logistic Regression Classifier: \n\n', classification_report(y_test, pred))

# Random Forest
xgb = RandomForestClassifier()
xgb.fit(x_train, y_train)
pred = xgb.predict(x_test)
print('Random Forest Classifier: \n\n', classification_report(y_test, pred))

#Linear Regression
regr = LinearRegression().fit(x_train, y_train)
y_hat = regr.predict(x_test)
y_hat[y_hat > 10] = 10
y_hat[y_hat < 0] = 0
y_hat = y_hat.round().astype("int")
print("Linear Regression Accuracy Score: {accur:.2f}".format(accur = accuracy_score(y_test, y_hat)))

#Decision Tree
dt = DecisionTreeClassifier(criterion="entropy", random_state=1).fit(x_train, y_train)
y_hat = dt.predict(x_test)
print("Decision Tree Accuracy Score: {accur:.2f}".format(accur = accuracy_score(y_test, y_hat)))


importances = xgb.feature_importances_
sorted_idx = importances.argsort()

# Crating a Data Frame with the feartures in the training set
feature_names = x_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Creating the Feature importance Table
plt.figure(figsize=(12, 6))
plt.barh(feature_importance_df['Feature'],importances[sorted_idx])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')

plt.show()