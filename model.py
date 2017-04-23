#import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
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
NUMBER_OF_ITERATIONS = 100

# import data
def import_data():
    data=pd.read_table('Complied-Data.txt', sep='\t', delimiter=None, delim_whitespace=False, header=0, index_col=0)
    X = (data.iloc[0:100, 1:50]) #NB: 150 is feature 1:149 for now, because of the long running time
    y = data.iloc[0:100,0]
    feature_names=(list(data))
    le.fit(y)
    y=le.transform(y)
    return(X,y, feature_names)

# returns trainset and testset
def split_data():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    return(X_train, X_test, y_train, y_test)
    
def recursive_feature_elimination_cv(X_train,y_train, X_test, y_test):
    #This function performs recursive feature selection with cross validation in a linear SVC model. 
      #It returns the selected features and the accuracy of the model in the test data set. 
    
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=KFold(10), scoring='accuracy')
    rfecv.fit(X_train, y_train)
    Nfeatures =rfecv.n_features_
    # Plot number of features VS. cross-validation scores
    #plt.figure()
    #plt.xlabel("Number of features selected")
    #plt.ylabel("Cross validation score (nb of correct classifications)")
    #plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    #plt.show()

    #This will give the ranking of the features
    RankFeatures=rfecv.ranking_
    
    #Determine the accuracy of the SVC model on the test-data 
    accuracy=rfecv.score(X_test, y_test)
    
    return [rfecv, accuracy, Nfeatures, RankFeatures]

def random_forest(X_train,y_train, X_test, y_test):
    feature_selection = SelectFromModel(SVC(kernel="linear"))
    classification = RandomForestClassifier()
    clf = Pipeline([
            ('feature_selection', feature_selection),
            ('classification', classification)])
    cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    clf = clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    Nfeatures = classification.n_features_
    RankFeatures=classification.feature_importances_
    
    return [clf, accuracy, Nfeatures, RankFeatures]

def accuracy_and_features(acc_rfecv, acc_rf, f_rfecv, f_rf):
    accuracy_list = [acc_rfecv, acc_rf]
    features_list = [f_rfecv, f_rf]
    names = ['Recursive Feature Elimination', 'Random Forest']
    for i in range(len(accuracy_list)):
        print('The average accuracy for %s is: %.3f +/- %.3f'  % (names[i], np.mean(accuracy_list[i]), np.std(accuracy_list[i])))
        print('The average used number of features for %s is: %.3f +/- %.3f' % (names[i], np.mean(features_list[i]), np.std(features_list[i])))
    
    count = 0
    for i in accuracy_list:
        i.sort(reverse=True)
        plt.plot(i, label= names[count])
        plt.legend(loc='upper right')
        if count == 0:
            highest_rfecv = i[0]
        elif count == 1:
            highest_rf = i[0]
        else:
            print('error')
        count+=1
    
    plt.xlabel('iteration')
    plt.ylabel('Accuracy')
    plt.show()
    
    if highest_rfecv > highest_rf:
        best_method, highest_accuracy = 'rfecv', highest_rfecv
    elif highest_rfecv < highest_rf:
        best_method, highest_accuracy = 'rf', highest_rf
    else:
        best_method, highest_accuracy = 'equal', highest_rfecv
    
    mean_rfecv = np.mean(acc_rfecv)
    mean_rf = np.mean(acc_rf)
    std_rfecv = np.std(acc_rfecv)
    std_rf = np.std(acc_rf)
    bar_plot_accuracy(mean_rfecv, mean_rf, std_rfecv, std_rf)
    return best_method, highest_accuracy

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
    ax.legend((rects1[0], rects2[0]), ('Recursive Feature Elimination', 'Random Forest'))
    plt.ylim((0,1))
    plt.show()
    
def save_best_model(best_method, result_rfecv, result_rf):
    if best_method == 'rfecv':
        models_list = result_rfecv
    else:
        models_list = result_rf
    models_list = sorted(models_list, key=itemgetter(1), reverse=True)
    #save best model
    joblib.dump(models_list[0][0], 'model.pkl')

def accuracy_versus_features(accuracy_rfecv, accuracy_rf, Nfeatures_rfecv, Nfeatures_rf):
    
    rfecv = plt.scatter(Nfeatures_rfecv, accuracy_rfecv, marker='x', color='b')
    rf = plt.scatter(Nfeatures_rf, accuracy_rf, marker='o', color='g')
    plt.xlabel('Number of used features')
    plt.ylabel('Accuracy')
    plt.legend((rfecv, rf), ('Recursive Feature Elimination', 'Random Forest') , loc='upper right')
    plt.show()

    Nfeatures_rfecv.sort() 
    Nfeatures_rf.sort()
    frequency_plots(Nfeatures_rfecv, Nfeatures_rf)
    
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
            plt.title('Frequency of the used number of features for Recursive Feature Elimination')
        else:
            plt.title('Frequency of the used number of features for Random Forest')
        plt.show()
    
def most_important_features(rfecv_list, rf_list):
    new_list = []
    important_features_rfecv = []
    global feature_names
    for i in rfecv_list:
        count = 1
        for feature in i[3]:
            new_list.append([feature, feature_names[count]])
            count += 1
        new_list.sort(reverse=True)
    important_features_rfecv.append(new_list[0:26])
    merged = list(chain(*important_features_rfecv))
    merged.sort(reverse=True)
    rank = []
    region = []
    for iteration in merged:
        rank.append(iteration[0])
        region.append(iteration[1])  
    c = Counter(region)
    c=c.most_common()
    #c.sort(key=itemgetter(1))
    labels, values = zip(*c)
    indexes = np.arange(len(labels))
    width = 0.5
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels, rotation = 'vertical')
    plt.xlabel('Chromosome region')
    plt.ylabel('Frequency of feature as part of the top 25 features with the highest ranking')
    plt.title('Frequency of the most important features for Recursive Feature Elimination')
    plt.show()

def create_output(rfecv_list, rf_list):
    accuracy_rfecv = [item[1] for item in rfecv_list]
    accuracy_rf = [item[1] for item in rf_list]
    Nfeatures_rfecv = [item[2] for item in rfecv_list]
    Nfeatures_rf = [item[2] for item in rf_list]
    
    most_important_features(rfecv_list, rf_list)
    best_method, highest_accuracy = accuracy_and_features(accuracy_rfecv, accuracy_rf, Nfeatures_rfecv, Nfeatures_rf)   
    save_best_model(best_method, rfecv_list, rf_list)
    accuracy_versus_features(accuracy_rfecv, accuracy_rf, Nfeatures_rfecv, Nfeatures_rf)
    
# main
X,y, feature_names = import_data()
rfecv_list = []
rf_list = []

for i in range(NUMBER_OF_ITERATIONS):
    X_train, X_test, y_train, y_test = split_data()
    rfecv_list.append(recursive_feature_elimination_cv(X_train, y_train, X_test, y_test))
    rf_list.append(random_forest(X_train,y_train, X_test, y_test))
    print('finished round %d out of %d' %  (i+1, NUMBER_OF_ITERATIONS))


create_output(rfecv_list, rf_list)


#TO DO: Calcalate average rank of the features 

#TO DO in a new script (see Templates of run_model Scirpts):
# Predict the labels of the new data set (with rfecv.predict(newdata)) for each of the 100 created models 
# and calculate how many times a sample is assigned to each group. Assign sample to group with the highest value
