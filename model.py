#import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from operator import itemgetter
from sklearn import preprocessing
from collections import Counter
le = preprocessing.LabelEncoder()

#Variables
TEST_SIZE = 0.10
NUMBER_OF_ITERATIONS = 3

# import data
def import_data():
    data=pd.read_table('Complied-Data.txt', sep='\t', delimiter=None, delim_whitespace=False, header=0, index_col=0)
    X = (data.iloc[0:100, 1:50])
    y = data.iloc[0:100,0]
    feature_names=(list(data))
    le.fit(y)
    y=le.transform(y)
    return(X,y, feature_names)

# returns trainset and testset
def split_data():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    return(X_train, X_test, y_train, y_test)

# SVM-RFE    
def recursive_feature_elimination_cv(X_train,y_train, X_test, y_test):
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=KFold(10), scoring='accuracy')
    rfecv.fit(X_train, y_train)
    # Determine the accuracy of the SVC model on the test-data, get the used number of features and ranking of the importance of features
    accuracy=rfecv.score(X_test, y_test)
    RankFeatures=rfecv.ranking_
    Nfeatures =rfecv.n_features_
    return [rfecv, accuracy, Nfeatures, RankFeatures]

# RF-SVM
def random_forest(X_train,y_train, X_test, y_test):
    feature_selection = SelectFromModel(SVC(kernel="linear"))
    classification = RandomForestClassifier()
    #Creating  pipeline for feature selection and classification
    clf = Pipeline([
            ('feature_selection', feature_selection),
            ('classification', classification)])
    cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    clf = clf.fit(X_train, y_train)
    # Determine the accuracy of the SVC model on the test-data, get the used number of features and ranking of the importance of features
    accuracy = clf.score(X_test, y_test)
    Nfeatures = classification.n_features_
    RankFeatures=classification.feature_importances_
    return [clf, accuracy, Nfeatures, RankFeatures]

#plot accuracy versus iteration and print accuracy and used number of features
def accuracy_and_features(acc_rfecv, acc_rf, f_rfecv, f_rf):
    accuracy_list = [acc_rfecv, acc_rf]
    features_list = [f_rfecv, f_rf]
    names = ['SVM-RFE', 'RF-SVM']
    for i in range(len(accuracy_list)):
        print('The average accuracy for %s is: %.3f +/- %.3f'  % (names[i], np.mean(accuracy_list[i]), np.std(accuracy_list[i])))
        print('The average used number of features for %s is: %.3f +/- %.3f' % (names[i], np.mean(features_list[i]), np.std(features_list[i])))
    
    count = 0
    for i in accuracy_list:
        i.sort(reverse=True)
        plt.plot(i, label= names[count])
        plt.legend(loc='upper right')
        count+=1
    #plot accuracy versus iteration    
    plt.xlabel('iteration')
    plt.ylabel('Accuracy')
    plt.show()

#makes a plot of the accuracy+std of both methods
def bar_plot_accuracy(mean_rfecv, mean_rf, std_rfecv, std_rf):
    fig, ax = plt.subplots()
    N=1
    width = 0.5
    ind = np.arange(N)
    rects1 = ax.bar(ind, mean_rfecv, width, color='b', yerr=std_rfecv)
    rects2 = ax.bar(ind + width, mean_rf, width, color='g', yerr=std_rf)
    
    ax.set_ylabel('Accuracy')
    xTickMarks = (('', ''))
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)
    ax.legend((rects1[0], rects2[0]), ('SVM-RFE', 'RF-SVM'))
    plt.ylim((0,1))
    plt.show()
   
#sort the list of accuracy values and picks/save the model with the median accuracy
def save_best_model(models_list, accuracy_rfecv):
    models_list = sorted(models_list, key=itemgetter(1), reverse=True)
    number = NUMBER_OF_ITERATIONS/2
    joblib.dump(models_list[int(number)][0], 'model.pkl')
    print(models_list[int(number)][0])

#plots the accuracy versus the used number of features of both methods
def accuracy_versus_features(accuracy_rfecv, accuracy_rf, Nfeatures_rfecv, Nfeatures_rf):
    rfecv = plt.scatter(Nfeatures_rfecv, accuracy_rfecv, marker='x', color='b')
    rf = plt.scatter(Nfeatures_rf, accuracy_rf, marker='o', color='g')
    plt.xlabel('Number of used features')
    plt.ylabel('Accuracy')
    plt.legend((rfecv, rf), ('SVM-RFE', 'RF-SVM') , loc='upper right')
    plt.show()

    Nfeatures_rfecv.sort() 
    Nfeatures_rf.sort()
    frequency_plots(Nfeatures_rfecv, Nfeatures_rf)

#plots the frequency of the used number of feautures for both methods    
def frequency_plots(Nfeatures_rfecv, Nfeatures_rf):
    method = [Nfeatures_rfecv, Nfeatures_rf]
    for i in method:
        labels, values = zip(*Counter(i).items())
        indexes = np.arange(len(labels))
        width = 0.5    
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.xlabel('Number of used features')
        plt.ylabel('Frequency')
        if i == Nfeatures_rfecv:
            plt.title('Frequency of the used number of features for SVM-RFE')
        else:
            plt.title('Frequency of the used number of features for RF-SVM')
        plt.show()

#plots the most important features    
def most_important_features(method_list, method_name):
    new_list = []
    important_features = []
    global feature_names
    for i in method_list:
        count = 1
        for feature in i[3]:
            new_list.append([feature, feature_names[count]])
            count += 1
        new_list.sort(reverse=True)
        important_features.append(new_list[0:25])
    merged = list(chain(*important_features))
    merged.sort(reverse=True)
    rank = []
    region = []
    for iteration in merged:
        rank.append(iteration[0])
        region.append(iteration[1])  
    c = Counter(region)
    c=c.most_common()
    labels, values = zip(*c)
    indexes = np.arange(len(labels))
    width = 0.5
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels, rotation = 'vertical')
    plt.xlabel('Feature')
    plt.ylabel('Frequency of 25 highest ranked features')
    plt.title('Frequency of the most important features for %s' % (method_name))
    plt.show()

#create some output like accuracy and number of features and execute different functions for figures
def create_output(rfecv_list, rf_list):
    accuracy_rfecv = [item[1] for item in rfecv_list]
    accuracy_rf = [item[1] for item in rf_list]
    Nfeatures_rfecv = [item[2] for item in rfecv_list]
    Nfeatures_rf = [item[2] for item in rf_list]
    
    most_important_features(rfecv_list, 'SVM-RFE')
    most_important_features(rf_list, 'RF-SVM')
    accuracy_and_features(accuracy_rfecv, accuracy_rf, Nfeatures_rfecv, Nfeatures_rf)   
    save_best_model(rfecv_list, accuracy_rfecv)
    accuracy_versus_features(accuracy_rfecv, accuracy_rf, Nfeatures_rfecv, Nfeatures_rf)
    
# main
# import data
X,y, feature_names = import_data()
rfecv_list = []
rf_list = []

# Repeat model making
for i in range(NUMBER_OF_ITERATIONS):
    X_train, X_test, y_train, y_test = split_data()
    rfecv_list.append(recursive_feature_elimination_cv(X_train, y_train, X_test, y_test))
    rf_list.append(random_forest(X_train,y_train, X_test, y_test))
    print('finished round %d out of %d' %  (i+1, NUMBER_OF_ITERATIONS))

# function for creating figures, print data ans save model
create_output(rfecv_list, rf_list)
