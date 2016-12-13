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
from subprocess import check_call
import pydot
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
sns.set(style='whitegrid', context='notebook')


cols = ['Redirect',
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

cm = np.corrcoef(df[cols].values.T) 
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()

result = np.zeros([28,2])
important_features = np.array(df.columns[:-1])
important_features1 = np.array(df.columns[:-1])

for i in range(0,28):
    print i,"\n"
    #print important_features
    X = df[important_features].values

    y = df[['Result']].values

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5, random_state=0)
    n = 100
    m = 2
    clf = RandomForestClassifier(n_estimators=n,criterion="entropy",max_features=m)

    Forest = clf.fit(X_train, y_train)
    predictions_rtree = clf.predict(X_test)
    
    accuracy = 100.0 * accuracy_score(y_test, predictions_rtree)
    print "The accuracy of your decision tree on testing data is: " + str(accuracy)
    print
    result[i][0] = len(important_features)
    result[i][1] = accuracy
    important_features = []
    print "min_importance:",np.min(Forest.feature_importances_)
    for x in range(0,len(Forest.feature_importances_)):
        #print "Hello2"
        min = np.min(Forest.feature_importances_)
        if Forest.feature_importances_[x] > min:
            print i,x
            #print "Hello1"
            important_features.append(str(important_features1[x]))
        else:
            #print "Hello"
            print x,df.columns[x]
        
    print 'Most important features:',important_features,"\n"
    print 
    print
    important_features1 = important_features

print result
plt.figure(1)
plt.subplot(211)
plt.plot(result[:,0], result[:,1], 'bo',result[:,0], result[:,1],  'k')
plt.axis([30, 2, 50, 100])
plt.show()

