'''
Script trains and evaluates the machine learning models for the crop clssification for a given ndvi values over the monthly average 

Model Evaluation: Logistic, SVM, KNN, Randon Forest, Gaussian Navies Bayes Algorithms 

Author: Srikanth, Shafik

'''


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

#Read the data file 
df = pd.read_csv("Croptype_ML.csv")
print(df)


#Interpolate to fill missing values
res = df.interpolate()

X = res.iloc[:, :-1].values
y = res.iloc[:, -1].values

#Define Imports 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

def run_exps(X_train: pd.DataFrame , y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    '''
    Lightweight script to test many models and find winners
    :param X_train: training split
    :param y_train: training target vector
    :param X_test: test split
    :param y_test: test target vector
    :return: DataFrame of predictions
    '''
        
    dfs = []
    models = [
            ('LogReg', LogisticRegression()), 
            ('RF', RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = 1)),
            ('KNN', KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2, leaf_size=10)),
            ('SVM', SVC(kernel = 'linear', random_state = 0, C = 3)), 
            ('GNB', GaussianNB())
            ]
    results = []
    names = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    target_names = ['apple', 'fallow', 'alfalfa hay', 'potato']
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=21)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(name)
        print(classification_report(y_test, y_pred, target_names=target_names))
        results.append(cv_results)
        names.append(name)
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
        final = pd.concat(dfs, ignore_index=True)
    return final

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

#Test and train split for model training and evaluation 
final = run_exps(X_train, y_train, X_test, y_test)
bootstraps = []
for model in list(set(final.model.values)):
    model_df = final.loc[final.model == model]
    bootstrap = model_df.sample(n=30, replace=True)
    bootstraps.append(bootstrap)
        
    bootstrap_df = pd.concat(bootstraps, ignore_index=True)
    results_long = pd.melt(bootstrap_df,id_vars=['model'],var_name='metrics', value_name='values')
    time_metrics = ['fit_time','score_time'] # fit time metrics
    ## PERFORMANCE METRICS
    results_long_nofit = results_long.loc[~results_long['metrics'].isin(time_metrics)] # get df without fit data
    results_long_nofit = results_long_nofit.sort_values(by='values')
    ## TIME METRICS
    results_long_fit = results_long.loc[results_long['metrics'].isin(time_metrics)] # df with fit data
    results_long_fit = results_long_fit.sort_values(by='values')

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20, 12))
sns.set(font_scale=2.5)
g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_nofit, palette="Set3")
plt.ylim(0, 1.5)
plt.legend(bbox_to_anchor=(0.55, 1), loc=2, borderaxespad=0.)
plt.title('Comparison of Model by Classification Metric')
# plt.savefig('./benchmark_models_performance.png',dpi=300)


plt.figure(figsize=(20, 12))
sns.set(font_scale=2.5)
g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_fit, palette="Set3")
plt.legend(bbox_to_anchor=(0.01, 1), loc=2, borderaxespad=0.)
plt.title('Comparison of Model by Fit and Score Time')
# plt.savefig('./benchmark_models_time.png',dpi=300)

plt.show()

metrics = list(set(results_long_nofit.metrics.values))
bootstrap_df.groupby(['model'])[metrics].agg([np.std, np.mean])


time_metrics = list(set(results_long_fit.metrics.values))
bootstrap_df.groupby(['model'])[time_metrics].agg([np.std, np.mean])


