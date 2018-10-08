#!/usr/bin/python

import sys
sys.path.append("../tools/")
sys.path.append("./modules/")

import pickle
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=4)

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# data_cleaning methods are defined in ./modules/data_cleaning.py
from data_cleaning import correct_data
from data_cleaning import data_info
from data_cleaning import process_data
from data_cleaning import create_value_feature
from data_cleaning import create_salary_bonus_feature

# ML_methods methods are defined in ./modules/ML_methods.py
from ML_methods import plot_EDA
from ML_methods import outliers_regression
from ML_methods import outliers_regression_plot
from ML_methods import outliers_identification
from ML_methods import get_kbest_features
from ML_methods import perform_classification
from ML_methods import grid_search

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi',\
                 'salary',\
                 'to_messages',\
                 'deferral_payments',\
                 'total_payments',\
                 'exercised_stock_options',\
                 'bonus',\
                 'restricted_stock',\
                 'shared_receipt_with_poi',\
                 'restricted_stock_deferred',\
                 'total_stock_value',\
                 'expenses',\
                 'loan_advances',\
                 'from_messages',\
                 'other',\
                 'from_this_person_to_poi',\
                 'director_fees',\
                 'deferred_income',\
                 'long_term_incentive',\
                 'from_poi_to_this_person']

# Loading the data into the data_dict dictionary
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "ENRON data loaded"
print

### Task 2: Remove outliers/fix records

# Outliers identification and EDA
# Just by looking at the data_dict keys we see that two keys: 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK'
# do not belong to any individual and most probably will be outliers
# First check of the data shows a mismatch in the records of two employees: BHATNAGAR SANJAY and BELFER ROBERT 

# The data is corrected in these two entries
data_dict = correct_data(data_dict)
print "BHATNAGAR SANJAY and BELFER ROBERT data corrected"
print

# Basic data information: number of features, etc...

n_people, n_features, n_pois, n_no_pois, nans_person, nans_feature = data_info(data_dict, features_list)

print "Basic information of the ENRON dataset"
print "Number of people   =", n_people
print "Number of features =", n_features
print "Number of POIs     =", n_pois

fig = plt.figure()
ax = fig.add_subplot(111)

nans_values_sorted = sorted(nans_feature.values())[::-1]
nans_keys_sorted   = []
keys_nans = list(nans_feature.keys())
for value_sorted in nans_values_sorted:
    for key in keys_nans:
        if value_sorted == nans_feature[key]:
            nans_keys_sorted.append(key)
            keys_nans.remove(key)
            break

ax.bar(nans_keys_sorted, nans_values_sorted) 
ax.set_xticklabels(nans_keys_sorted,rotation=90)
plt.title("Number of missing values per feature")
ax.set_xlabel("Feature name")
ax.set_ylabel("Missing values")
plt.text(0,1.06,'Fig. 1',\
        horizontalalignment='left',\
        verticalalignment='baseline',\
        weight = 'semibold',\
        fontsize = 12,\
        transform = ax.transAxes)
fig.tight_layout()
plt.savefig("./report/figures/features_nans.pdf")

## Data exploration

# As we will see later, these features are the most important in the ENRON dataset
features_plot = ['poi', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options']
data_raw, data_clean, labels_clean, features_clean = process_data(data_dict, features_plot)
plot_EDA(data_raw, data_clean, features_plot)

print "raw vs clean data plot saved to ./figures/plot_EDA.pdf"
print

# 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK' are removed from the dataset
for key in ["TOTAL", "THE TRAVEL AGENCY IN THE PARK"]:
    data_dict.pop(key, 0)

# Further outlier search with the "regression method"
features_salary_bonus = ['poi', 'salary', 'bonus']
features_stock        = ['poi', 'total_stock_value', 'exercised_stock_options']

regression_plot_outliers    = dict.fromkeys(['salary_bonus', 'stock'], 0)
regression_plot_no_outliers = dict.fromkeys(['salary_bonus', 'stock'], 0)
r2_plot_outliers    = dict.fromkeys(['salary_bonus', 'stock'], 0)
r2_plot_no_outliers = dict.fromkeys(['salary_bonus', 'stock'], 0)
print "EDA performed in 'salary', 'bonus', 'total_stock_value' and 'exercised_stock_options' features\n"
for f, r in zip([features_salary_bonus, features_stock], ['salary_bonus', 'stock']):
    data_raw, data_clean, labels_clean, features_clean = process_data(data_dict, f)

    data_outliers, data_plot_outliers, data_plot_no_outliers, r2_outliers, r2_no_outliers = outliers_regression(features_clean, f)

    regression_plot_outliers[r]    = data_plot_outliers
    regression_plot_no_outliers[r] = data_plot_no_outliers
    r2_plot_outliers[r]    = r2_outliers
    r2_plot_no_outliers[r] = r2_no_outliers
    
    outliers_names = outliers_identification(data_dict, data_outliers, f)
    print "Outliers in ", f
    print outliers_names
    for name in outliers_names:
        print data_dict[name]['poi'],"  ",name
    print

print "Plotting outliers regression\n"
outliers_regression_plot(regression_plot_outliers, regression_plot_no_outliers, r2_plot_outliers, r2_plot_no_outliers)

# With KBest() method the 10 best features are identified

scores, features_out = get_kbest_features(data_dict, features_list)
print "KBest features scores"
print
for i,j in features_out:
    print "|", i, "|", np.round(j,3), "|"
print

### Task 3: Create new feature(s)

# total_value feature is calculated with the sum of the payments and the stock value

data_dict = create_value_feature(data_dict)
data_dict = create_salary_bonus_feature(data_dict)
print "New feature 'total_value' created"
print
print "Performing again KBest with new feature"
features_list = features_list + ['total_value'] + ['salary_bonus']
scores, features_out = get_kbest_features(data_dict, features_list)
print "KBest features scores ('total_value' and 'salary_bonus' included)"
print
for i,j in features_out:
    print "|", i, "|", np.round(j,3), "|"
print

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
feature_selection = SelectKBest(k = 10)

# Naive Bayes classifier

print "Performing Naive Bayes classification\n"
clf_NB   = GaussianNB()
steps_NB = [('feature_selection', feature_selection), ('Naive_Bayes', clf_NB)]
pipe_NB, results_NB = perform_classification(clf_NB, steps_NB, features, labels)

# Decision Tree Classifier

print "Performing Decission Tree classification\n"
clf_DT   = DecisionTreeClassifier()
steps_DT = [('feature_selection', feature_selection), ('Decission_Tree', clf_DT)]
pipe_DT, results_DT = perform_classification(clf_DT, steps_DT, features, labels)

# Random Forest classifier

print "Performing Random Forest classification\n"
clf_RF   = RandomForestClassifier()
steps_RF = [('feature_selection', feature_selection), ('Random_Forest', clf_RF)]
pipe_RF, results_RF = perform_classification(clf_RF, steps_RF, features, labels)

# Ada Boost classifier

print "Performing Ada Boost classification\n"
clf_AB   = AdaBoostClassifier()
steps_AB = [('feature_selection', feature_selection), ('Ada_Boost', clf_AB)]
pipe_AB, results_AB = perform_classification(clf_AB, steps_AB, features, labels)

classifiers = ['NB', 'DT', 'RF', 'AB']
results     = [results_NB, results_DT, results_RF, results_AB]
for c, r in zip(classifiers, results):
    print c+" classifier"
    for score in r.keys():
        print score, "=", r[score]
    print

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print "Tuning parameters of classifiers"
print

# Naive Bayes classifier

print "Performing Grid Search of Naive Bayes classification"
param_dict_NB = {'feature_selection__k': range(5, len(features_list))}
gs = grid_search(steps_NB, param_dict_NB, features, labels)
gs_clf_NB = gs.best_estimator_
print '\n Score Metrics Decission Tree Classifier'
test_classifier(gs_clf_NB, data_dict, features_list, folds = 1000)
print

# The rest of the parameter tuning is comented out due to time execution
# NB turned out to be the classifier with better scores

'''
# Decision Tree classifier

print "Performing Grid Search of Decission Tree classification"
param_dict_DT = {'feature_selection__k': range(5, len(features_list)),\
                 'Decission_Tree__criterion': ['gini', 'entropy'],\
                 'Decission_Tree__min_samples_split' : [2, 3, 4, 5, 6, 7, 8, 9, 10]}
gs = grid_search(steps_DT, param_dict_DT, features, labels)
gs_clf_DT = gs.best_estimator_
print '\n Score Metrics Decission Tree Classifier'
test_classifier(gs_clf_DT, data_dict, features_list, folds = 100)
print

# Random Forest classifier

print "Performing Grid Search of Random Forest classification"
param_dict_RF = {'feature_selection__k': range(5, len(features_list)),\
                 'Random_Forest__n_estimators': range(5,15),\
                 'Random_Forest__criterion': ['gini', 'entropy'],\
                 'Random_Forest__min_samples_split' : [2, 3, 4, 5, 6, 7, 8, 9, 10]}
gs = grid_search(steps_RF, param_dict_RF, features, labels)
gs_clf_DT = gs.best_estimator_
print '\n Score Metrics Random Forest Classifier'
test_classifier(gs_clf_DT, data_dict, features_list, folds = 100)
print

# AdaBoost classifier

print "Performing Grid Search of AdaBoost classification"

param_dict_AB = {'feature_selection__k': range(1, len(features_list)),\
                 'Ada_Boost__learning_rate': np.linspace(0.2,2,10),\
                 'Ada_Boost__algorithm': ['SAMME', 'SAMME.R']}
gs = grid_search(steps_AB, param_dict_AB, features, labels)
gs_clf_AB = gs.best_estimator_
print '\n Score Metrics AdaBoost Classifier'
test_classifier(gs_clf_AB, data_dict, features_list, folds = 100)
print
'''

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = gs_clf_NB
dump_classifier_and_data(clf, data_dict, features_list)

