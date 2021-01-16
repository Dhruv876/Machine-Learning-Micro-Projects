# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 01:11:08 2019

@author: dhruv
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 25
fig_size[1] = 15
plt.rcParams["figure.figsize"] = fig_size

data = pd.read_csv("C://Users//dhruv//OneDrive//Desktop//stats project//HCV-Egy-Data.csv")

X = data.iloc[:,0:28]
y = data.iloc[:,27:29]

y = y.drop(['Baseline histological Grading'], axis=1)

bins_target = [0, 2, 4]
target_groups = [0, 1]
y['target'] = pd.cut(y['Baselinehistological staging'], bins_target, labels=target_groups)
y = y.drop(['Baselinehistological staging'], axis=1)

corrmat = data.corr()
top_corr_features = corrmat.index

#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
column = pd.DataFrame(X.columns)
a = np.zeros([28,2])
for i in range(0,28):
    x = data.iloc[:,i]
    z = data.iloc[:,-1]
    selection = pearsonr(x, z)
    a[i] = selection
    
b = np.concatenate((column, a), axis=1)


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features

bins_wbc = [0, 7534, 12101]
bins_rbc = [0, 4422130, 5018451]
bins_hgb = [0, 13, 20]
bins_alt1 = [0, 84, 128]
bins_alt24 = [0, 35, 128]
bins_rnab = [0, 590951, 1201086]
bins_rna12 = [0, 288754, 3731527]
#bins_rnaef = [0, 291378, 808450]

wbc_groups = [1, 2]
rbc_groups = [1, 2]
hgb_groups = [1, 2]
alt1_groups = [1, 2]
alt24_groups = [1, 2]
rnab_groups = [1, 2]
rna12_groups = [1, 2]
#rnaef_groups = [1, 2]

X['wbc'] = pd.cut(X['WBC'], bins_wbc, labels=wbc_groups)
X['rbc'] = pd.cut(X['RBC'], bins_rbc, labels=rbc_groups)
X['hgb'] = pd.cut(X['HGB'], bins_hgb, labels=hgb_groups)
X['alt1'] = pd.cut(X['ALT 1'], bins_alt1, labels=alt1_groups)
X['alt24'] = pd.cut(X['ALT after 24 w'], bins_alt24, labels=alt24_groups)
X['rnab'] = pd.cut(X['RNA Base'], bins_rnab, labels=rnab_groups)
X['rna12'] = pd.cut(X['RNA 12'], bins_rna12, labels=rna12_groups)
#X['rnaef'] = pd.cut(X['RNA EF'], bins_rnaef, labels=rnaef_groups)

X = X.drop(['RNA Base', 'RNA EF', 'Age ','BMI','Fever','Headache ','Diarrhea ','Epigastric pain ', 'HGB', 'Plat','AST 1','ALT4','ALT 12','ALT 24','ALT 36','ALT 48','RNA 4','RNA EOT','Baseline histological Grading', 'WBC', 'RBC', 'ALT 1', 'ALT after 24 w', 'RNA Base', 'RNA 12', 'RNA EF'], axis=1)

#print(X.head())

Y = y.to_numpy()
Y = np.ravel(Y)

#train model!
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

classifier_1 = RandomForestClassifier(n_estimators=10)
boost = AdaBoostClassifier(base_estimator=classifier_1, n_estimators=400, learning_rate=1)
model = boost.fit(X_train, y_train)
predicted_1 = model.predict(X_test)
score_1 = model.score(X_test, y_test)
print(r'Random Forest with boosting accuracy = ', score_1)

classifier_2 = RandomForestClassifier(n_estimators=10)
classifier_2.fit(X_train, y_train)
predicted_2 = classifier_2.predict(X_test)
score_2 = classifier_2.score(X_test, y_test)
print(r'Random Forest without boosting accuracy = ', score_2)

classifier_3 = LogisticRegression()
boost_2 = AdaBoostClassifier(base_estimator=classifier_3, n_estimators=400, learning_rate=1)
boost_2.fit(X_train, y_train)
predicted_3 = boost_2.predict(X_test)
score3 = boost_2.score(X_test, y_test)
print(r'Logistic reg with boosting accuracy = ', score3)

classifier_4 = GaussianNB()
classifier_4.fit(X_train,y_train)
predicted_4 = classifier_4.predict(X_test)
score_4 = classifier_4.score(X_test, y_test)
print(r'Logistic reg without boosting accuracy = ', score_4)

#cross Validation

scores_randomforest = cross_val_score(boost, X_train, y_train, cv=10,error_score='raise-deprecating')
print(r'average score of rand for = ', np.mean(scores_randomforest))

scores_logisticreg = cross_val_score(classifier_4, X_train, y_train, cv=10,error_score='raise-deprecating')
print(r'average score of log reg = ', np.mean(scores_logisticreg))

#so GNB gives high accuracy as compared to random forest also there is no effect of boosting on logistic regression algo

#ROC curve

fpr1, tpr1, thresholds1 = roc_curve(y_test,predicted_1)
area_under_curve1 = auc(fpr1,tpr1)

fpr2, tpr2, thresholds2 = roc_curve(y_test,predicted_3)
area_under_curve2 = auc(fpr2,tpr2)

fpr3, tpr3, thresholds3 = roc_curve(y_test,predicted_4)
area_under_curve3 = auc(fpr3,tpr3)

plt.subplot(131)
plt.plot(fpr1,tpr1)
plt.title(r'area under roc curve= %0.2f' % (area_under_curve1))

plt.subplot(132)
plt.plot(fpr2,tpr2)
plt.title(r'area under roc curve= %0.2f' % (area_under_curve2))

plt.subplot(133)
plt.plot(fpr3,tpr3)
plt.title(r'area under roc curve= %0.2f' % (area_under_curve3))
