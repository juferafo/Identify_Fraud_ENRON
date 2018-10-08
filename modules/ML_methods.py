#!/usr/bin/python

import sys
sys.path.append("../tools/")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from feature_format import featureFormat, targetFeatureSplit
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate 
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_kbest_features(data_dict, features_list):
    '''
    This module performs a select_kbest of num_features and returns a dictionary 
    containing the score and name of the selected features.
    '''
    
    if len(features_list) == 0:
        print "ERROR: Empty features_list!"
        return None

    data = featureFormat(data_dict, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    selector = SelectKBest()
    #selector.fit(features, labels)
    selector.fit_transform(features, labels)
    scores  = selector.scores_

    indices = range(1,len(scores)+1)
    scores = np.column_stack((indices, scores))
    scores = scores[scores[:,1].argsort()][::-1]
    
    features_out = {}
    for i in range(len(scores)):
        features_out[features_list[int(scores[i,0])]] = scores[i,1]
    
    features_out = sorted(features_out.items(), key=lambda kv: kv[1], reverse=True)

    return scores, features_out


def plot_EDA(data_raw, data_clean, features_list):
    '''
    This plots the data of the provided features.
    It is assumed that the data contains the information of the following features:

    features_plot = ['poi', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options']

    '''

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
   
    for i, j in zip(range(len(data_clean)), range(len(data_raw))):
        legend_poi = None

        if data_raw[j,0] == 1:
            color_raw = 'red'
        elif data_raw[j,0] == 0:
            color_raw = 'blue'
        else:
            color_raw = 'k'
        
        if data_clean[j,0] == 1:
            color_clean = 'red'
        elif data_clean[j,0] == 0:
            color_clean = 'blue'
        else:
            color_clean = 'k'

        ax1.plot(data_raw[j,1], data_raw[j,2], '.', color = color_raw)
        ax2.plot(data_clean[i,1], data_clean[i,2], '.', color = color_clean)
        ax3.plot(data_raw[j,3], data_raw[j,4], '.', color = color_raw)
        ax4.plot(data_clean[i,3], data_clean[i,4], '.', color = color_clean)

    ax1.set_title("Data with outliers")
    ax2.set_title("Data without outliers")
  
    legend = [ Line2D([0], [0], marker='o', color = 'white', markerfacecolor='red', label='POIs'),\
               Line2D([0], [0], marker='o', color='white', markerfacecolor='blue', label='Non POIs')]

    ax1.legend(handles=legend, loc = 2)
    
    ax1.set_xlabel(features_list[1]+" ($)")
    ax1.set_ylabel(features_list[2]+" ($)")
    ax2.set_xlabel(features_list[1]+" ($)")
    ax2.set_ylabel(features_list[2]+" ($)")
    
    ax3.set_xlabel(features_list[3]+" ($)")
    ax3.set_ylabel(features_list[4]+" ($)")
    ax4.set_xlabel(features_list[3]+" ($)")
    ax4.set_ylabel(features_list[4]+" ($)")

    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax4.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    '''
    plt.text(-0.25,1.06,'Fig. 2',\
             horizontalalignment='left',\
             verticalalignment='baseline',\
             weight = 'semibold',\
             fontsize = 12,\
             transform = ax1.transAxes)
    '''
    fig.tight_layout()
    plt.savefig("./report/figures/plot_EDA.pdf")


def outliers_identification(data_dict, data_outliers, features_list):
    '''
    This method identifies the names of the outliers detected in outliers_regression
    '''
    names_outliers = []
    for k_name in data_dict.keys():
        if data_dict[k_name][features_list[1]] in list(data_outliers[:,0]):
            names_outliers.append(k_name)

    return names_outliers


def outliers_regression(features, features_list, outlier_removal = 0.1):
    '''
    This method filters the data and indentifies the present outliers.
    It performs a linear regression and computes the difference between the
    predicted model an the real values. Those with bigger difference are the outliers

    It returns the name of the outliers in each plot

    This method is implemented to work in one single 2D plot with 2 features!
    '''

    features = np.asarray(features)
    
    x_outliers = features[:,0]
    y_outliers  = features[:,1]

    x_outliers = np.reshape( np.array(x_outliers), (len(x_outliers), 1))
    y_outliers  = np.reshape( np.array(y_outliers), (len(y_outliers), 1))

    reg_out = LinearRegression()
    reg_out.fit(x_outliers, y_outliers)
    y_predictions_outliers = reg_out.predict(x_outliers)
    
    mat_diff = np.column_stack((x_outliers, y_outliers, np.abs(y_outliers - y_predictions_outliers)))
    mat_diff = mat_diff[mat_diff[:,2].argsort()][::-1]
    len_out = int(len(mat_diff)*outlier_removal)
    data_outliers = mat_diff[:len_out,:]
    mat_diff = mat_diff[len_out:,:]
    
    x = np.reshape( np.array(mat_diff[:,0]), (len(mat_diff[:,0]), 1))
    y  = np.reshape( np.array(mat_diff[:,1]), (len(mat_diff[:,1]), 1))
    
    reg = LinearRegression()
    reg.fit(x, y)
    y_predict = reg.predict(x)
   
    r2_outliers    = r2_score(y_outliers, y_predictions_outliers)
    r2_no_outliers = r2_score(y, y_predict)

    data_plot_outliers    = np.column_stack((x_outliers, y_outliers, y_predictions_outliers))
    data_plot_no_outliers = np.column_stack((x, y, y_predict))

    return data_outliers, data_plot_outliers, data_plot_no_outliers, r2_outliers, r2_no_outliers
    

def outliers_regression_plot(regression_plot_outliers, regression_plot_no_outliers, r2_plot_outliers, r2_plot_no_outliers):
    '''
    This method plots the data and regression calculated in outliers_regression for both sets of data:
    bonus vs salary and exercised_stock_options vs total_stock_value
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.set_title("Outlier removal by regression")

    for k, ax in zip(['salary_bonus', 'stock'], [ax1, ax2]):
        ax.plot(regression_plot_outliers[k][:,0], regression_plot_outliers[k][:,1], '.', color = 'b')
        ax.plot(regression_plot_no_outliers[k][:,0], regression_plot_no_outliers[k][:,1], '.', color = 'r')

        ax.plot(regression_plot_outliers[k][:,0], regression_plot_outliers[k][:,2],\
                'b-', label = "R2 outliers = "+str(r2_plot_outliers[k].round(4)))
        ax.plot(regression_plot_no_outliers[k][:,0], regression_plot_no_outliers[k][:,2],\
                'r-', label = "R2 no outliers = "+str(r2_plot_no_outliers[k].round(4)))
        
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    ax1.set_xlabel('salary ($)')
    ax1.set_ylabel('bonus ($)')
    ax2.set_xlabel('total_stock_value ($)')
    ax2.set_ylabel('exercised_stock_options ($)')
    ax2.legend(loc = 4) 
    ax1.legend(loc = 4) 
    fig.tight_layout()
   
    plt.savefig("./report/figures/outlier_removal_regression.pdf")

    return None


def perform_classification(clf, steps, features, labels, folds = 1000):
    '''
    This method performs the cross validation using the provided clf classifier.
    It builds a Pipeline with the selected steps. 
    It returns both the estimator and a dictionary containing 
    the mean value of the accuracy, precission, recall and f1 scores.
    '''

    cv_kfold = StratifiedShuffleSplit(labels, folds, random_state = 42)
    pipe  = Pipeline(steps)
    
    scoring = {'accuracy'  : make_scorer(accuracy_score), 
               'precision' : make_scorer(precision_score),
               'recall'    : make_scorer(recall_score), 
               'f1_score'  : make_scorer(f1_score)}

    results = cross_validate(estimator = pipe,\
                             X = features,\
                             y = labels,\
                             cv = cv_kfold,\
                             scoring = scoring)
   
    for key in results.keys():
        results[key] = np.mean(results[key])

    return pipe, results


def grid_search(steps, clf_parameters, features, labels, folds = 1000):
    '''
    This method performs a cross validation parameter tuning.
    To do so, it builds a Pipeline and employes GridSearchCV over the provided
    range of parameters.
    '''

    pipe    = Pipeline(steps)
    cv_kfold = StratifiedShuffleSplit(labels, folds, random_state = 42)
    cv_grid = GridSearchCV(pipe, param_grid = clf_parameters, cv = cv_kfold)
    cv_grid.fit(features, labels)

    return cv_grid
