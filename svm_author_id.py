#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# try kernel linear and rbf, optimize C parameter
clf = SVC(kernel='rbf', C=10000)

# smaller the training set divide it by 100, reduce training time
features_train = features_train[:len(features_train) / 100]
labels_train = labels_train[:len(labels_train) / 100]

# train model and calculate time
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time() - t0, 3), "s"

# test model and calculate time
t1 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time() - t1, 3), "s"

# get your result and print
accuracy = accuracy_score(labels_test, pred)
print("Accuracy is equal to %0.4F %%" % (accuracy * 100))

# print the 10, 26 nad 50 row of the dataset's preiction
print(pred[10])
print(pred[26])
print(pred[50])

# count the number of the prediction is 1
count = 0
for i in pred:
    if i == 1:
        count = count + 1
print count
#########################################################
