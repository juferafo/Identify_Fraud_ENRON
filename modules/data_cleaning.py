#!/usr/bin/python

import sys
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

def correct_data(data_dict):
    '''
    This method corrects the data for the employees BHATNAGAR SANJAY and BELFER ROBERT.
    It returns the updated data_dict dictionary according to the values found in ./enron61702insiderpay.pdf
    '''
    
    for i in ['BHATNAGAR SANJAY', 'BELFER ROBERT']:
        data_dict[i] = data_dict.fromkeys(data_dict[i].keys(), 'NaN')
    
    data_robert = {'deferred_income': -102500,\
                   'expenses': 3825,\
                   'director_fees': 102500,\
                   'total_payments': 3285,\
                   'restricted_stock': 44093,\
                   'restricted_stock_deferred': -44093,\
                   'poi': False}
    
    data_sanjay = {'expenses': 137864,\
                   'total_payments': 137864,\
                   'exercised_stock_options': 15456290,\
                   'restricted_stock': 2604490,\
                   'restricted_stock_deferred': -2604490,\
                   'total_stock_value': 15456290,\
                   'from_messages': 29,\
                   'to_messages': 523,\
                   'from_this_person_to_poi': 1,\
                   'from_poi_to_this_person': 0,\
                   'shared_receipt_with_poi': 463,\
                   'poi': False}

    for user, dict_values in zip(['BHATNAGAR SANJAY', 'BELFER ROBERT'], [data_sanjay, data_robert]):
        for key in dict_values:
            data_dict[user][key] = dict_values[key]
    
    return data_dict


def process_data(data, features_list):
    '''
    This method returns the data_raw data_clean, features, labels of the given data_dict.
    data_raw is the original data with outliers
    data_clean is the data without the 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK' entries
    features and labels are calculated from data_clean

    The featureFormat and targetFeatureSplit are employed to obtain the aforementioned variables
    By default sort_keys remove_any_zeroes are set to True
    '''
    
    data_raw = featureFormat(data, features_list, sort_keys = True, remove_any_zeroes = True)
    data_valid = {}
    for i in data.keys():
        if i not in ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']:
            data_valid[i] = data[i]

    data_clean = featureFormat(data_valid, features_list, sort_keys = True, remove_any_zeroes = True)
    labels_clean, features_clean = targetFeatureSplit(data_clean)

    return data_raw, data_clean, labels_clean, features_clean


def data_info(data_dictionary, features_list):
    '''
    This method returns the main parameters of the dataset such as:

    n_people     = Number of people in the dataset
    n_pois       = Number of POIs
    n_no_pois    = Number of non POIs
    n_features   = Number of features used
    nans_person  = dictionary with the number of NaNs per person
    nans_feature = dictionary with the number of NaNs per feature
    '''
    
    valid_people = list(data_dictionary.keys())
    for i in ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']:
        valid_people.remove(i)
    
    n_people   = len(valid_people)
    # 'poi' is considered a label not a feature!
    n_features = len(features_list) - 1

    n_pois    = 0
    n_no_pois = 0
    nans_feature = dict.fromkeys(features_list, 0)
    nans_person  = dict.fromkeys(data_dictionary.keys(), 0)
    for k_name in valid_people:
        if data_dictionary[k_name]['poi'] == 1:
            n_pois += 1
        elif data_dictionary[k_name]['poi'] == 0:
            n_no_pois += 1
        else:
            # Just in case there are missing values in the poi labels
            pass

        for k_feature in data_dictionary[k_name].keys():
            if data_dictionary[k_name][k_feature] == 'NaN':
                nans_person[k_name] += + 1
                if k_feature in features_list:
                    nans_feature[k_feature] += 1
    
    return n_people, n_features, n_pois, n_no_pois, nans_person, nans_feature


def create_value_feature(data_dict):
    '''
    This method creates a new feature called 'total_value'.
    'total_value' is calculated with the sum of 'salary', 'bonus' and 'total_stock_value'

        'total_value' = 'salary' + 'bonus' + 'total_stock_value'
    '''
    
    for key in data_dict.keys():
        # Since we are using the data_dict which main contain NaN values in different features
        # it is necessary to have defined values in 'salary', 'bonus' and 'total_stock_value'
        if data_dict[key]['salary'] != 'NaN' and data_dict[key]['bonus'] != 'NaN' and \
                data_dict[key]['total_stock_value'] != 'NaN':

            data_dict[key]['total_value'] = data_dict[key]['bonus'] + data_dict[key]['salary'] + \
                    data_dict[key]['total_stock_value']
        # The rest is considered NaN
        else:
            data_dict[key]['total_value'] = 'NaN'

    return data_dict


def create_salary_bonus_feature(data_dict):
    '''
    This method creates a new feature called 'salary_bonus'.
    'salary_bonus' is calculated with the sum of 'salary' and 'bonus'

        'salary_bonus' = 'salary' + 'bonus'
    '''
    
    for key in data_dict.keys():
        # Since we are using the data_dict which main contain NaN values in different features
        # it is necessary to have defined values in 'salary' and 'bonus'
        if data_dict[key]['salary'] != 'NaN' and data_dict[key]['bonus'] != 'NaN':
            data_dict[key]['salary_bonus'] = data_dict[key]['bonus'] + data_dict[key]['salary']
        # The rest is considered NaN
        else:
            data_dict[key]['salary_bonus'] = 'NaN'

    return data_dict
