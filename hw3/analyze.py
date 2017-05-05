# Building an ML Pipeline, CAPP 30254
#
# Scripts to run the magic loop to classify and evalute different models
# 
# Ratul Esrar


from __future__ import division
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import time
import pylab as pl


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import tree, svm
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# From Assignment 2
def logistic_model(X_train, X_test, y_train):
    '''
    Function to select best features using RFE, then fits logistic regression model. Returns predicted values.

    Inputs:
    	X_train, X_test, y_train (df)

    Output:
    	predicted_y (list)
    '''
    reg = LogisticRegression()
    rfe = RFE(reg)
    rfe = rfe.fit(X_train, y_train)
    predicted_y = rfe.predict(X_test)
    best_features = rfe.get_support(indices = True)

    return predicted_y, best_features

# Magicloop code adapted from Rayid Ghani: https://github.com/rayidghani/magicloops/blob/master/magicloops.py
def LR():
    return LogisticRegression(penalty = 'l1', C = 1e5)


def KNN():
    return KNeighborsClassifier(n_neighbors = 3)


def DT():
    return DecisionTreeClassifier()


def SVM():
    return svm.SVC(kernel = 'linear', probability = True, random_state = 3)


def RF():
    return RandomForestClassifier(n_estimators = 50, n_jobs = -1)


def AB():
    return AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                                    algorithm="SAMME",
                                                    n_estimators=200)


def GB():
    return GradientBoostingClassifier(learning_rate = 0.05,
                                    	subsample = 0.5,
                                    	max_depth = 6,
                                    	n_estimators = 10)


def NB():
    return GaussianNB()


def define_clfs_params_test(grid_size = 'test'):
    clfs = {'LR':LR(),'KNN':KNN(),'DT':DT(),'SVM':SVM(),'RF':RF(),'AB':AB(),'GB':GB(),'NB':NB()}

    test_grid = {'LR':{'penalty':['l1'], 'C':[0.01]},
        'KNN':{'n_neighbors':[5], 'weights':['uniform'], 'algorithm':['auto']},
        'DT':{'criterion':['gini'], 'max_depth':[1], 'max_features':['sqrt'], 'min_samples_split':[10]},
        'SVM':{'C':[0.01], 'kernel':['linear']},
        'RF':{'n_estimators':[1], 'max_depth':[1], 'max_features':['sqrt'], 'min_samples_split':[10]},
        'AB':{'algorithm':['SAMME'], 'n_estimators':[1]},
        'GB':{'n_estimators':[1], 'learning_rate':[0.1], 'subsample':[0.5], 'max_depth':[1]},
        'NB':{},
           }

    small_grid = {'LR':{'penalty':['l1','l2'], 'C':[0.00001,0.001,0.1,1,10]},
        'KNN':{'n_neighbors':[1,5,10,25,50,100], 'weights':['uniform','distance'], 'algorithm':['auto','ball_tree','kd_tree']},
        'DT':{'criterion':['gini', 'entropy'], 'max_depth':[1,5,10,20,50,100], 'max_features':['sqrt','log2'], 'min_samples_split':[2,5,10]},
        'SVM':{'C':[0.00001,0.0001,0.001,0.01,0.1,1,10], 'kernel':['linear']},
        'RF':{'n_estimators':[10,100], 'max_depth':[5,50], 'max_features':['sqrt','log2'],'min_samples_split':[2,10]},
        'AB':{ 'algorithm':['SAMME', 'SAMME.R'], 'n_estimators':[1,10,100,1000,10000]},
        'GB':{'n_estimators':[10,100], 'learning_rate':[0.001,0.1,0.5], 'subsample':[0.1,0.5,1.0], 'max_depth':[5,50]},
        'NB':{},
           }

    large_grid = {'LR':{'penalty':['l1','l2'], 'C':[0.00001,0.0001,0.001,0.01,0.1,1,10]},
        'KNN':{'n_neighbors':[1,5,10,25,50,100], 'weights':['uniform', 'distance'], 'algorithm':['auto','ball_tree','kd_tree']},
        'DT':{'criterion':['gini', 'entropy'], 'max_depth':[1,5,10,20,50,100], 'max_features':['sqrt','log2'], 'min_samples_split':[2,5,10]},
        'SVM':{'C':[0.00001,0.0001,0.001,0.01,0.1,1,10], 'kernel':['linear']},
        'RF':{'n_estimators':[1,10,100,1000,10000], 'max_depth':[1,5,10,20,50,100], 'max_features':['sqrt','log2'], 'min_samples_split':[2,5,10]},
        'AB':{'algorithm':['SAMME', 'SAMME.R'], 'n_estimators':[1,10,100,1000,10000]},
        'GB':{'n_estimators':[1,10,100,1000,10000], 'learning_rate':[0.001,0.01,0.05,0.1,0.5], 'subsample':[0.1,0.5,1.0], 'max_depth':[1,3,5,10,20,50,100]},
        'NB':{},
           }

    if grid_size == 'test':
        return clfs, test_grid
    elif grid_size == 'small':
        return clfs, small_grid
    elif grid_size == 'large':
        return clfs, large_grid
    else:
        return 0, 0


def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary


def precision_at_k(y_true, y_scores, k):
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    return precision


def plot_precision_recall_n(y_true, y_prob, model_name):
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    # plt.savefig(name)
    plt.show()


def clf_loop(models_to_run, clfs, grid, X, y):
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc','p_at_5', 'p_at_10', 'p_at_20'))
    for n in range(1, 2):
        # create training and valdation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    # you can also store the model, feature importances, and prediction scores
                    # we're only storing the metrics for now
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0)]
                    plot_precision_recall_n(y_test,y_pred_probs,clf)
                except IndexError:
                    print('Error: ',e)
                    continue
    

    return results_df

