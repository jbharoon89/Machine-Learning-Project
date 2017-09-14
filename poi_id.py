
# coding: utf-8

# In[1]:

#!/usr/bin/python

# POI_ID.PY -> Modified as per final project requirements.
# Done by Javeed Basha

import sys
import pickle
import numpy
import tester

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from numpy import mean

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score



# Task 1: Selecting what features to use.
# -> features_list is a list of strings, each of which is a feature name.
# -> The first feature must be "poi".
# -> You will need to use more features
# -> Since the first feature must be 'poi - person of interest'
# -> I am using a separate label for it.

# Mandatory POI label feature -> ['poi']

poi_label = ['poi']

# FIN_EMAIL_LIST(Finance & Email) 
# -> 20 Features including Mandatory Feature!!

fin_email_list = ['salary',
                  'bonus',
                  'deferral_payments',
                  'deferred_income',
                  'director_fees',
                  'exercised_stock_options',
                  'expenses',
                  'loan_advances',
                  'long_term_incentive',
                  'other',
                  'restricted_stock',
                  'restricted_stock_deferred',
                  'total_payments',
                  'total_stock_value',
                  'from_messages',
                  'from_poi_to_this_person',
                  'from_this_person_to_poi',
                  'shared_receipt_with_poi',
                  'to_messages']

# Features List

features_list = poi_label + fin_email_list

# Printing the Features List

print "Features List:", features_list
print "\n Length:", len(features_list)

# Load the dictionary containing the dataset
# -> Dataset - final_project_dataset.pkl

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
# -> Removing Outliers using name.pop('')
# -> A total of 3 outliers have been removed.
# -> LOCKHART EUGENE E is removed as per suggestion.

data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
data_dict.pop("LOCKHART EUGENE E")

# Task 3: Create new feature(s)

# -> Creating a new feature 'total_sum'
# -> This feature contains total sum of the finance feature field.

fields = ['salary', 'bonus', 'total_stock_value', 'exercised_stock_options']
for value in data_dict:
    inf = data_dict[value]
    is_null = False
    for feature in fields:
        if inf[feature] == 'NaN':
            is_null = True
    if not is_null:
        inf['total_sum'] = inf['salary'] + inf['bonus'] + inf['total_stock_value'] + inf['exercised_stock_options'] 
    else:
        inf['total_sum'] = 'NaN'

features_list += ['total_sum']

# -> Creating a new feature 'email_percentage_to_poi'
# -> This features calculates total percentages of mails
# -> sent to 'poi'

fields = ['to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi']
for value in data_dict:
    inf = data_dict[value]
    is_valid = True
    for feature in fields:
        if inf[feature] == 'NaN':
            is_valid = False
        if is_valid:
            total_messages = inf['to_messages'] + inf['from_messages']
            poi_messages = inf['from_poi_to_this_person'] + inf['from_this_person_to_poi']
            inf['email_percentage_to_poi'] = float(poi_messages) / total_messages
        else:
            inf['email_percentage_to_poi'] = 'NaN'

features_list += ['email_percentage_to_poi']

# Updated features_list and total_length

print " \n Updated Features List:", features_list
print " \n Updated Length:", len(features_list)

# Store to my_dataset for easy export below.

my_dataset = data_dict

# Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# Creating a MinMaxScaler()
# -> Transforms features by scaling each feature to a given range.
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# Get K-Best (Selecting 10 Best Values)
# -> Using Sklearn's SelectKBest() feature selection
# -> It returns a dictionary.
   
k_best = SelectKBest(f_classif, k=10)
k_best.fit(features, labels)

best_fea_list = zip(k_best.get_support(), features_list[1:], k_best.scores_)
best_fea_list = sorted(best_fea_list, key=lambda x: x[2], reverse=True)

print "\n K-best 10 features:", best_fea_list[:10]

# Using only the ten best features + [poi]
# -> All the sample training and test data 
# -> will be created using the given features
# -> selected using KBest Feature Selection Algorithm.

best_features = ['poi',
                 'exercised_stock_options',
                 'total_stock_value',
                 'bonus',
                 'salary',
                 'total_sum',
                 'deferred_income',
                 'long_term_incentive',
                 'restricted_stock',
                 'total_payments',
                 'shared_receipt_with_poi']

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, best_features)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                            test_size=0.3, 
                                                                            random_state=60)

# Parameter tuning using GridSearchCV
# -> Please name your classifier clf for easy export below.
# -> Pipelines - http://scikit-learn.org/stable/modules/pipeline.html

### Parameters: -> To obtain best parameters!!!
def clf_parameters(grid_search, features, labels, parameters):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                                test_size=0.3,
                                                                                random_state=60)
    grid_search.fit(features_train, labels_train)
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t %s=%r, ' % (param_name, best_parameters[param_name])

### Supervised Algorithm

# Non-Ordered Algorithm -> Naive Bayes
gauss_clf = GaussianNB()
parameters = {}
grid_search = GridSearchCV(gauss_clf, parameters)
clf_parameters(grid_search, features, labels, parameters)

# Non-Ordered Algorithm -> Decision Tree
des_clf = tree.DecisionTreeClassifier()
parameters = {'criterion': ['gini', 'entropy'],
              'min_samples_split': [2, 10, 20],
              'max_depth': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              'max_leaf_nodes': [5, 10, 20]}
grid_search = GridSearchCV(des_clf, parameters)
print '\n Decision Tree Best Parameters:'
clf_parameters(grid_search, features, labels, parameters)

# Non-Ordered Algorithm -> AdaBoost
ada_clf = AdaBoostClassifier()
parameters = {'n_estimators': [50,100,200], 'algorithm': ['SAMME', 'SAMME.R'],
              'learning_rate': [0.4,0.6,1]}
grid_search = GridSearchCV(ada_clf, parameters)
print '\n Ada Boost Best Parameters:'
clf_parameters(grid_search, features, labels, parameters)

# Random Forest Classifier Algorithm
rf_clf = RandomForestClassifier()
parameters = {'n_estimators': [10, 20, 30], 'n_jobs': [1, -1],
              'random_state': [42, 50, 60]}
grid_search = GridSearchCV(rf_clf, parameters)
print '\n Random Forest Best Parameters:'
clf_parameters(grid_search, features, labels, parameters)

### Unsupervised Algorithm
# KMeans Cluster
kmean_clf = KMeans()
parameters = {'n_clusters': [2], 'algorithm': ['elkan', 'full'], 'tol': [1e-2, 1e-3, 1e-4]}
grid_search = GridSearchCV(kmean_clf, parameters)
print '\n KMeans Best Parameters:'
clf_parameters(grid_search, features, labels, parameters)

# Task 4: Try a varity of classifiers
# -> Please name your classifier clf for easy export below.
# -> Pipelines - http://scikit-learn.org/stable/modules/pipeline.html
# -> Using the best parameters obtained, I am trying a variety of classifiers.
# -> These classifier's are tested using the test_classifier()

# Task 5: Tune your classifier to achieve better than .3 precision and recall 
# -> using our testing script. Check the tester.py script in the final project
# -> folder for details on the evaluation method, especially the test_classifier
# -> function. Because of the small size of the dataset, the script uses

# Non-Ordered Algorithm -> Naive Bayes
g_clf = GaussianNB()

print '\n Gaussian Naive Bayes Output:', tester.test_classifier(g_clf, my_dataset, best_features)

# Non-Ordered Algorithm -> Decision Tree
d_clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 2, max_leaf_nodes = 5,
                                    min_samples_leaf = 10, min_samples_split = 2)

print '\n Decision Tree Output:', tester.test_classifier(d_clf, my_dataset, best_features)

# Non-Ordered Algorithm -> AdaBoost
a_clf = AdaBoostClassifier(n_estimators=50, algorithm = 'SAMME', learning_rate = 0.4)

print '\n Ada Boost Output:', tester.test_classifier(a_clf, my_dataset, best_features)

# Random Forest Classifier Algorithm
r_clf = RandomForestClassifier(n_estimators = 10, n_jobs = 1, random_state = 50)

print '\n Random Forest Output:', tester.test_classifier(r_clf, my_dataset, best_features)

### Unsupervised Algorithm
# KMeans Cluster
# k_clf = KMeans(n_clusters = 2, algorithm = 'elkan', tol = 0.001)

print '\n KMeans Output:', tester.test_classifier(k_clf, my_dataset, best_features)

### Selecting Best Algorithm to dump the data:

clf = g_clf

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# -> check your results. You do not need to change anything below, but make sure
# -> that the version of poi_id.py that you submit can be run on its own and
# -> generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, best_features)


# In[ ]:



