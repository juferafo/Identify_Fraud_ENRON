# Udacity Data Analysis Nanodegree project
# Identify fraud from ENRON email

The goal of this project is to identify persons of interest among the employees of ENRON, where POI stands for any individual 
who was responsible for any financial irregularity that led to the corporation's insolvency.
To do so, we analyze the dataset `./final_project_dataset.pkl` that contains financial and email-related information of the employees.
This analysis is performed employing machine learning techniques where predictive 
model is trained to classify the employees into POIs or non-POIs.
We also test and discuss the classification efficiency of different algorithms such as Naive Bayes, AdaBoost or Random Forest classifiers.

### Contents

This work was developed with `Python` using the library `scikit-learn`. 
The main script `./poi_id.py` imports the user-defined libraries in `./modules/`
to process and classify the data. 

The project report can be found in `./report/identify_fraud_ENRON_email.html` or `./report/identify_fraud_ENRON_email.ipynb`.
