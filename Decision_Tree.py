# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from sklearn import cross_validation
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import svm, cross_validation, metrics, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from graphviz import Digraph

df = pd.read_csv('/Users/riyasuchdev/Downloads/data.csv')
df.columns = ['having_IP_Address',
'URL_Length',
'Shortining_Service',
'having_At_Symbol',
'double_slash_redirecting',
'Prefix_Suffix',
'having_Sub_Domain',
'SSLfinal_State',
'Domain_registeration_length',
'Favicon',
'port',
'HTTPS_token',
'Request_URL',
'URL_of_Anchor',
'Links_in_tags',
'SFH',
'Submitting_to_email',
'Abnormal_URL',
'Redirect',
'on_mouseover',
'RightClick',
'popUpWidnow',
'Iframe',
'age_of_domain',
'DNSRecord',
'web_traffic ',
'Page_Rank',
'Google_Index',
'Links_pointing_to_page',
'Statistical_report',
'Result']

df.head()

cols = ['Prefix_Suffix',
'having_Sub_Domain',
'SSLfinal_State',
'URL_of_Anchor',
'Links_in_tags','web_traffic ']

cols1 = ['Redirect',
'Shortining_Service',
'double_slash_redirecting',
'SSLfinal_State',
'Favicon',
'port',
'HTTPS_token',
'URL_of_Anchor',
'Submitting_to_email',
'Abnormal_URL',
'on_mouseover',
'RightClick',
'popUpWidnow',
'Iframe',
'Result'] 
"""
sns.set(style='whitegrid', context='notebook')
sns.pairplot(df[cols], size=1.5);
plt.show()

cm = np.corrcoef(df[cols].values.T) 
sns.set(font_scale = 1.5)
#hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
#plt.show()"""

X = df[df.columns[:-1]].values

y = df[['Result']].values

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5, random_state=0)

print"===========tree=============================\n"
#classifier_tree = tree.DecisionTreeClassifier()
#classifier_tree = tree.DecisionTreeClassifier(max_depth=7)
#classifier_tree = tree.DecisionTreeClassifier(criterion="entropy")
classifier_tree = tree.DecisionTreeClassifier(criterion="entropy",max_depth=4)
#Decision tree classifier created.

#Beginning model training.
# Train the decision tree classifier
classifier_tree.fit(X_train, y_train)
#print "Model training completed."

# Use the trained classifier to make predictions on the test data
predictions_tree = classifier_tree.predict(X_test)
#Predictions on testing data computed.

# Print the accuracy (percentage of phishing websites correctly predicted)
accuracy = 100.0 * accuracy_score(y_test, predictions_tree)
print "The accuracy of your decision tree on testing data is: " + str(accuracy)

#create decision tree and save it in tree1.dot and open it using the website mentioned in th
tree.export_graphviz(classifier_tree, out_file = '/Users/riyasuchdev/Downloads/tree6.dot')
print

importances = classifier_tree.feature_importances_

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[f]))
    #print("%d. feature %d (%f) %s" % (f + 1,f+1, importances[f], df.columns[indices[f]]))
"""
print"===============Random forest classifier===============\n"

#clf1 = ExtraTreesClassifier(n_estimators=250)
clf1 = RandomForestClassifier(n_estimators=25,criterion="entropy",max_features=7)
#print "Beginning model training."

clf1.fit(X_train, y_train)
#print "Model training completed."

# Use the trained classifier to make predictions on the test data
predictions_etree = clf1.predict(X_test)
#print "Predictions on testing data computed."

# Print the accuracy (percentage of phishing websites correctly predicted)
accuracy = 100.0 * accuracy_score(y_test, predictions_etree)
print "The accuracy of your decision tree on testing data is: " + str(accuracy)
print

importances = clf1.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf1.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
"""

