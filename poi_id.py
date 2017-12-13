#!/usr/bin/python

import sys
import pickle
import numpy
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### initial exploratory data analysis

print "number of data points = ",len(data_dict.keys())

print "features in this dataset:", data_dict['METTS MARK'].keys()

print "number of features = ", len(data_dict['METTS MARK'].keys())

### feature selection. I will use SelectKBest here to find the initial best features.
### For now I will choose an arbitrary K = 6 value to find the best 6 features.
### During the tweaking and optimization process I will feed SelectKBest into GridSearchCV
### to find the optimal K value and redefine the features list as necessary

### remove the TOTAL row

data_dict.pop("TOTAL", 0)

flist = ['poi',
         'salary',
         'to_messages',
         'deferral_payments',
         'total_payments',
         'exercised_stock_options',
         'bonus', 'restricted_stock',
         'shared_receipt_with_poi',
         'restricted_stock_deferred',
         'total_stock_value',
         'expenses',
         'loan_advances',
         'from_messages',
         'other',
         'from_this_person_to_poi',
         'director_fees',
         'deferred_income',
         'long_term_incentive',
         'from_poi_to_this_person']

from feature_format import featureFormat, targetFeatureSplit
data1 = featureFormat(data_dict, flist, sort_keys=True)
labels1, features1 = targetFeatureSplit(data1)

from sklearn.feature_selection import SelectKBest
bf = SelectKBest(k=6)
best_features = bf.fit_transform(features1, labels1)

mask = bf.get_support()
KBestFeatures=[]
for i in range(len(mask)):
    if mask[i]==True:
        KBestFeatures.append(flist[i+1])
print "Best initial features = ", KBestFeatures

initial_features_list = ['salary', 
'exercised_stock_options', 
'bonus', 
'total_stock_value', 
'deferred_income', 
'long_term_incentive']

### Task 2: Remove outliers

### remove employees with more than 16 NaN values
### use pandas to convert data_dict to a dataframe for ease of outlier removal

import pandas as pd
df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))
df.set_index(employees, inplace=True)

nanlist=[]
for i in range(len(df.index)):
    counter=0
    for j in range(len(df.columns)):
        if df.loc[df.index[i]][df.columns[j]] == 'NaN':
            counter+=1
    nanlist.append(counter)
    
outlier_employee=[]
for i in range(len(nanlist)):
    if nanlist[i]>16:
        outlier_employee.append(employees[i])

print "Employees with greater than 16 NaN values:", outlier_employee

### remove outlier employees from data_dict

for i in outlier_employee:
	data_dict.pop(i, None)


### Task 3: Create new feature(s)

### I began thinking about what information is not relayed in the initial dataset. 
### I had a hunch that a person of interest may have a huge stock option in relation 
### to their salary so I used this as a starting point in creating my new feature. 
### I will create a new feature that takes the ratio of total stock options to salary. 
### Unfortunately there are a lot of entries that have a missing value for either 
### the salary or the total stock value. 
### I will filter these out and enter the value as zero to avoid a division error.

for i in data_dict.keys():
    if (data_dict[i]['total_stock_value']=='NaN') or (data_dict[i]['salary']=='NaN'):
        data_dict[i]['stock_salary_ratio']=0
    elif (data_dict[i]['total_stock_value']==0) or (data_dict[i]['salary']==0):
        data_dict[i]['stock_salary_ratio']=0
    else:
        data_dict[i]['stock_salary_ratio']=float(data_dict[i]['total_stock_value'])/float(data_dict[i]['salary'])

### Rescaling features
### I will rescale the features in my features list for use with the 
### support vector machine classifier with the rbf kernel. Technically I do not need to
### scale the initial selected features as it is all measured in dollars, but I plan on using created features
### that are not measured in dollars, so a rescaling will be necessary.

list_a = []
list_b = []
list_c = []
list_d = []
list_e = []
list_f = []

list_list = [list_a,list_b,list_c,list_d,list_e, list_f]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

for i in range(len(list_list)):
    for j in data_dict.keys():
        list_list[i].append(data_dict[j][initial_features_list[i]])
        list_list[i] = filter(lambda a: a!='NaN' , list_list[i])

list_a_new = []
list_b_new = []
list_c_new = []
list_d_new = []
list_e_new = []
list_f_new = []


### cast each item as a float

for item in list_list[0]:
    list_a_new.append(float(item))

for item in list_list[1]:
    list_b_new.append(float(item))
    
for item in list_list[2]:
    list_c_new.append(float(item))
    
for item in list_list[3]:
    list_d_new.append(float(item))
    
for item in list_list[4]:
    list_e_new.append(float(item))
    
for item in list_list[5]:
    list_f_new.append(float(item))




salarylist=[]
exercisedstocklist=[]
bonuslist=[]
totalstocklist=[]
deferredincomelist=[]
longtermincentivelist=[]

for i in range(len(list_a_new)):
    salary_list=[]
    salary_list.append(list_a_new[i])
    salarylist.append(salary_list)

for i in range(len(list_b_new)):
    exercisedstock_list=[]
    exercisedstock_list.append(list_b_new[i])
    exercisedstocklist.append(exercisedstock_list)

for i in range(len(list_c_new)):
    bonus_list=[]
    bonus_list.append(list_c_new[i])
    bonuslist.append(bonus_list)

for i in range(len(list_d_new)):
    totalstock_list=[]
    totalstock_list.append(list_d_new[i])
    totalstocklist.append(totalstock_list)
    
for i in range(len(list_e_new)):
    deferredincome_list=[]
    deferredincome_list.append(list_e_new[i])
    deferredincomelist.append(deferredincome_list)
    
for i in range(len(list_f_new)):
    longtermincentive_list=[]
    longtermincentive_list.append(list_f_new[i])
    longtermincentivelist.append(longtermincentive_list)
    
salarylist = numpy.array(salarylist)
exercisedstocklist = numpy.array(exercisedstocklist)
bonuslist = numpy.array(bonuslist)
totalstocklist = numpy.array(totalstocklist)
deferredincomelist = numpy.array(deferredincomelist)
longtermincentivelist = numpy.array(longtermincentivelist)

scaled_salary = scaler.fit_transform(salarylist)
scaled_exercised_stock = scaler.fit_transform(exercisedstocklist)
scaled_bonus = scaler.fit_transform(bonuslist)
scaled_total_stock = scaler.fit_transform(totalstocklist)
scaled_deferred_income = scaler.fit_transform(deferredincomelist)
scaled_longterm_incentive = scaler.fit_transform(longtermincentivelist)

rescaled_salary = []
rescaled_exercised_stock = []
rescaled_bonus = []
rescaled_total_stock = []
rescaled_deferred_income = []
rescaled_longterm_incentive = []

for i in range(len(list_a_new)):
    rescaled_salary.append(scaled_salary[i][0])
for i in range(len(list_b_new)):
    rescaled_exercised_stock.append(scaled_exercised_stock[i][0])
for i in range(len(list_c_new)):
    rescaled_bonus.append(scaled_bonus[i][0])
for i in range(len(list_d_new)):
    rescaled_total_stock.append(scaled_total_stock[i][0])
for i in range(len(list_e_new)):
    rescaled_deferred_income.append(scaled_deferred_income[i][0])
for i in range(len(list_f_new)):
    rescaled_longterm_incentive.append(scaled_longterm_incentive[i][0])

count = 0
for i in data_dict.keys():
    if data_dict[i]['salary']!='NaN':
        data_dict[i]['scaled_salary']=rescaled_salary[count]
        count += 1
    else:
        data_dict[i]['scaled_salary']=0
        
count1 = 0
for i in data_dict.keys():
    if data_dict[i]['exercised_stock_options']!='NaN':
        data_dict[i]['scaled_exercised_stock']=rescaled_exercised_stock[count1]
        count1 += 1
    else:
        data_dict[i]['scaled_exercised_stock']=0
        
count2 = 0
for i in data_dict.keys():
    if data_dict[i]['bonus']!='NaN':
        data_dict[i]['scaled_bonus']=rescaled_bonus[count2]
        count2 += 1
    else:
        data_dict[i]['scaled_bonus']=0
        
count3 = 0
for i in data_dict.keys():
    if data_dict[i]['total_stock_value']!='NaN':
        data_dict[i]['scaled_total_stock']=rescaled_total_stock[count3]
        count3 += 1
    else:
        data_dict[i]['scaled_total_stock']=0
        
count4 = 0
for i in data_dict.keys():
    if data_dict[i]['deferred_income']!='NaN':
        data_dict[i]['scaled_deferred_income']=rescaled_deferred_income[count4]
        count4 += 1
    else:
        data_dict[i]['scaled_deferred_income']=0
        
count5 = 0
for i in data_dict.keys():
    if data_dict[i]['long_term_incentive']!='NaN':
        data_dict[i]['scaled_longterm_incentive']=rescaled_longterm_incentive[count5]
        count5 += 1
    else:
        data_dict[i]['scaled_longterm_incentive']=0

### Store to my_dataset for easy export below.

my_dataset = data_dict

### Extract features and labels from dataset for local testing

features_list = ['poi',
'salary',
 'exercised_stock_options',
 'bonus',
 'total_stock_value',
 'deferred_income',
 'long_term_incentive',
'stock_salary_ratio']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# Use the train_test_split module to split our data into training and testing data

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# first classifier: Naive Bayes

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score
clf1 = GaussianNB()
pred1 = clf1.fit(features_train,labels_train).predict(features_test)
acc1 = accuracy_score(pred1, labels_test)
rec1 = recall_score(labels_test, pred1)
prec1 = precision_score(pred1, labels_test)
print "GaussianNB accuracy = ", acc1
print "GaussianNB recall score =", rec1
print "GaussianNB precision score =", prec1

# second classifier: Decision Tree

from sklearn import tree
clf2 = tree.DecisionTreeClassifier()
pred2 = clf2.fit(features_train, labels_train).predict(features_test)
acc2 = accuracy_score(labels_test, pred2)
rec2 = recall_score(labels_test, pred2)
prec2 = precision_score(labels_test, pred2)
print "Decision Tree accuracy = ", acc2
print "Decision Tree recall score =", rec2
print "Decision Tree precision score =", prec2

Scaled_Features = ['poi',
                    'scaled_salary',
                    'scaled_exercised_stock',
                    'scaled_bonus', 
                   'scaled_total_stock',
                    'scaled_deferred_income',
		    'scaled_longterm_incentive',
                    'stock_salary_ratio']

# rerun our cross validation with the new selected features
from sklearn.svm import SVC
data2 = featureFormat(data_dict, Scaled_Features)
labels2, features2 = targetFeatureSplit(data2)
features_train2, features_test2, labels_train2, labels_test2 = train_test_split(features2, labels2, test_size=0.3, random_state=42)

# Apply SVC

clf3 = SVC(kernel='rbf')
pred3 = clf3.fit(features_train2, labels_train2).predict(features_test2)
acc3 = accuracy_score(labels_test2, pred3)
rec3 = recall_score(labels_test2, pred3)
prec3 = precision_score(labels_test2, pred3)
print "SVC accuracy = ", acc3
print "SVC recall score =", rec3
print "SVC precision score =", prec3

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

data = featureFormat(my_dataset, flist, sort_keys = True)
labels, features = targetFeatureSplit(data)
kbest = SelectKBest()
sss = StratifiedShuffleSplit(n_splits = 10, test_size = 0.2, random_state=42)
dtree=tree.DecisionTreeClassifier(random_state=42)
pipe = Pipeline([('kbest',kbest),('dtree',dtree)])
K_FEATURES = [2,3,4,5,6]
param_grid = {
        'kbest__k':K_FEATURES,
         'dtree__criterion': ['gini','entropy'],
          'dtree__max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],
          }
grid_search = GridSearchCV(estimator = pipe,
                           scoring = 'f1',
                          param_grid=param_grid,
                          cv=sss)
best = grid_search.fit(features, labels)

print "Best parameters: ", grid_search.best_params_

bf1 = SelectKBest(k=grid_search.best_params_['kbest__k'])
best_features1 = bf1.fit_transform(features, labels)

mask1 = bf1.get_support()
KBestFeatures1=[]
for i in range(len(mask1)):
    if mask1[i]==True:
        KBestFeatures1.append(flist[i+1])
print "Best final features = ", KBestFeatures1

poi = ['poi']

features_list = poi + KBestFeatures1

clf = tree.DecisionTreeClassifier(criterion = grid_search.best_params_['dtree__criterion'], max_depth = grid_search.best_params_['dtree__max_depth'], random_state=42)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)