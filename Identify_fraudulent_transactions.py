# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 

train = pd.read_csv('.../input/transactionRecord.csv')
train.head()
train.describe()
print(train.groupby(['Class']).size())  # imbalanced dataset
Fraud_transacation = train[train["Class"]==1]
Normal_transacation = train[train["Class"]==0]
print("percentage of fraud transacation is", len(Fraud_transacation) / len(train) * 100)
print("percentage of normal transacation is", len(Normal_transacation) / len(train) * 100)

plt.figure(figsize=(15, 8))
plt.subplot(121)
Fraud_transacation.Amount.plot.hist(title="Fraud Transacation")
plt.subplot(122)
Normal_transacation.Amount.plot.hist(title="Normal Transaction")
plt.figure(figsize=(15, 8))
plt.subplot(121)
Fraud_transacation[Fraud_transacation["Amount"]<=2000].Amount.plot.hist(title="Fraud Tranascation")
plt.subplot(122)
Normal_transacation[Normal_transacation["Amount"]<=2000].Amount.plot.hist(title="Normal Transaction")

from sklearn.preprocessing import StandardScaler

# Normalization
train['Amount'] = StandardScaler().fit_transform(train['Amount'].values.reshape(-1, 1))
train['Time'] = StandardScaler().fit_transform(train['Amount'].values.reshape(-1, 1))

train.drop(['Id'], axis=1, inplace=True)
print(train.head())

### Undersampling the Majority Class
train = train.sample(frac=1) # shuffle the data
fraud_df = train.loc[train['Class'] == 1]
non_fraud_df = train.loc[train['Class'] != 1][:395]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# shuffle the rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)
print(new_df.shape)
new_df.head()

X = new_df.drop('Class', axis=1)
y = new_df['Class']
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import cross_val_score

classifiers = {'LogisticRegression': LogisticRegression(),
               'DecisionTreeClassifier': DecisionTreeClassifier(), 
               'RandomForestClassifier': RandomForestClassifier(),
               'AdaBoostClassifier': AdaBoostClassifier(), 
               'GradientBoostingClassifier': GradientBoostingClassifier()}

for key, classifier in classifiers.items():
    print(key)
    classifier.fit(X_train_under, y_train_under.values.ravel())
    training_score = cross_val_score(classifier, X_train_under, y_train_under.values.ravel(), cv=5, scoring='f1')
    print("Classifiers: ", classifier.__class__.__name__, "has a training score of", round(training_score.mean(), 2) * 100, "% f1 score")
    
from sklearn.model_selection import GridSearchCV

# Linear Regression Classifier
lr_params = {"penalty": ['l1', 'l2'], 
             'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_lr = GridSearchCV(LogisticRegression(), lr_params, scoring='f1')
grid_lr.fit(X_train_under, y_train_under.values.ravel())
param_lr = grid_lr.best_estimator_
print(grid_lr.best_params_)

# Decision Tree Classifier
dt_params = {"criterion": ["gini", "entropy"], 
             "max_depth": list(range(2,4,1)), 
             "min_samples_leaf": list(range(5,7,1))}
grid_dt = GridSearchCV(DecisionTreeClassifier(), dt_params, scoring='f1')
grid_dt.fit(X_train, y_train)

param_dt = grid_dt.best_estimator_
print(grid_dt.best_params_)

# Random Forest Classifier
rf_params = { 'n_estimators': [200, 500],
              'max_depth' : [4,5,6,7,8],
              'criterion' :['gini', 'entropy']}
grid_rf = GridSearchCV(RandomForestClassifier(), rf_params, scoring='f1')
grid_rf.fit(X_train_under, y_train_under.values.ravel())
param_rf = grid_rf.best_estimator_
print(grid_rf.best_params_)

# Adaboost Classifier
ab_params = { 'n_estimators': [50, 100],
              'learning_rate' : [0.01,0.05,0.1,0.3,1]}
grid_ab = GridSearchCV(AdaBoostClassifier(), ab_params, scoring='f1')
grid_ab.fit(X_train_under, y_train_under.values.ravel())
param_ab = grid_ab.best_estimator_
print(grid_ab.best_params_)

# Gradient Boosting Classifier
gb_params = { "learning_rate": [0.01,0.05,0.1,0.3,1],
              "max_features":["log2","sqrt"],
              "criterion": ["friedman_mse",  "mae"]}
grid_gb = GridSearchCV(GradientBoostingClassifier(), gb_params, scoring='f1')
grid_gb.fit(X_train_under, y_train_under.values.ravel())
param_gb = grid_gb.best_estimator_
print(grid_gb.best_params_)

param_lr_score = cross_val_score(param_lr, X_train_under, y_train_under.values.ravel(), cv=5, scoring='f1')
print('Logistic regression cross validation f1 score: ', round(param_lr_score.mean() * 100, 2).astype(str) + '%')

param_dt_score = cross_val_score(param_dt, X_train_under, y_train_under.values.ravel(), cv=5, scoring='f1')
print('Decision Tree cross validation f1 score: ', round(param_dt_score.mean() * 100, 2).astype(str) + '%')

param_rf_score = cross_val_score(param_rf, X_train_under, y_train_under.values.ravel(), cv=5, scoring='f1')
print('Random Forest cross validation f1 score: ', round(param_rf_score.mean() * 100, 2).astype(str) + '%')

param_ab_score = cross_val_score(param_ab, X_train_under, y_train_under.values.ravel(), cv=5, scoring='f1')
print('Adaboost cross validation f1 score: ', round(param_ab_score.mean() * 100, 2).astype(str) + '%')

param_gb_score = cross_val_score(param_gb, X_train_under, y_train_under.values.ravel(), cv=5, scoring='f1')
print('Gradient Boosting cross validation f1 score: ', round(param_gb_score.mean() * 100, 2).astype(str) + '%')

pred_under = param_ab.predict(X_test_under)
print(f1_score(y_test_under, pred_under))

### Oversampling the Minority Class
print(len(y_train[y_train['Class'] == 1]))
from imblearn.over_sampling import SMOTE


print("Before OverSampling, counts of label '1': {}".format(len(y_train[y_train['Class'] == 1])))
print("Before OverSampling, counts of label '0': {} \n".format(len(y_train[y_train['Class'] == 0])))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(X_train_res, y_train_res, test_size=0.2, random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report

lr_params = {"penalty": ['l1', 'l2'], 
             'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_lr_over = GridSearchCV(LogisticRegression(), lr_params)
grid_lr_over.fit(X_train_over, y_train_over)
param_lr_over = grid_lr_over.best_estimator_
print(grid_lr_over.best_params_)

# training set f1 score
param_lr_score_over = param_lr_over.predict(X_train_over)
print(classification_report(y_train_over, param_lr_score_over))
oversampling_train_f1 = f1_score(y_train_over, param_lr_score_over)
print(oversampling_train_f1)

# testing set f1 score
y_pre = param_lr_over.predict(X_test_over)
print(classification_report(y_test_over, y_pre))
oversampling_test_f1 = f1_score(y_test_over, y_pre)
print(oversampling_test_f1)
