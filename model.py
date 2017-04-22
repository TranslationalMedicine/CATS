import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

#Variables
TEST_SIZE = 0.10
NUMBER_OF_SAMPLES_CV = 90
NUMBER_OF_FEATURES_CV = 100
NUMBER_OF_CLASSES_CV = 3

# import data
def import_data():
    data=pd.read_table('Complied-Data.txt', sep='\t', delimiter=None, delim_whitespace=False, header=0, index_col=0)
    X = (data.iloc[0:100, 1:150]) #NB: 150 is feature 1:149 for now, because of the long running time
    y = data.iloc[0:100,0]
    return(X,y)

# returns train and test set
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
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
    rfecv.fit(X_train, y_train)
    Nfeatures =rfecv.n_features_
    # Plot number of features VS. cross-validation scores
    #plt.figure()
    #plt.xlabel("Number of features selected")
    #plt.ylabel("Cross validation score (nb of correct classifications)")
    #plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    #plt.show()
    
    # Get the selected features 
    features=rfecv.get_support(indices=True)
    #This will give the ranking of the features
    RankFeatures=rfecv.ranking_
    
    #Determine the accuracy of the SVC model on the test-data 
    accuracy=rfecv.score(X_test, y_test)
    
    return(features, accuracy, rfecv, RankFeatures, Nfeatures)

def random_forest(X_train,y_train, X_test, y_test):
    feature_selection = SelectFromModel(SVC(kernel="linear"))
    classification = RandomForestClassifier()
    clf = Pipeline([
            ('feature_selection', feature_selection),
            ('classification', classification)])
    accuracy = clf.fit(X_train, y_train).score(X_test, y_test)
    Nfeatures = classification.n_features_
    RankFeatures=classification.feature_importances_
    
    return clf, accuracy, Nfeatures,RankFeatures

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
        count += 1
    plt.xlabel('iteration')
    plt.ylabel('Accuracy')
    plt.show()
    
# main
X,y = import_data()
result_rfecv = {}
result_rf = {}
accuracy_list_rfecv =[]
accuracy_list_rf =[]
Nfeatures_list_rf = []
Nfeatures_list_rfecv = []

for i in range(5):
    X_train, X_test, y_train, y_test = split_data()
    features_rfecv, accuracy_rfecv, rfecv, rankfeatures_rfecv, Nfeatures_rfecv = recursive_feature_elimination_cv(X_train, y_train, X_test, y_test)
    rf, accuracy_rf, Nfeatures_rf, RankFeatures_rf = random_forest(X_train,y_train, X_test, y_test)
    result_rfecv[i] = [features_rfecv,accuracy_rfecv,rfecv]
    result_rf[i] = [accuracy_rf, rf]
    accuracy_list_rfecv.append(accuracy_rfecv)
    accuracy_list_rf.append(accuracy_rf)
    Nfeatures_list_rf.append(Nfeatures_rf)
    Nfeatures_list_rfecv.append(Nfeatures_rfecv)
    print('finished round;', i+1)
accuracy_and_features(accuracy_list_rfecv, accuracy_list_rf, Nfeatures_list_rf, Nfeatures_list_rfecv)

#TO DO: Calcalate average rank of the features 

#TO DO in a new script (see Templates of run_model Scirpts):
# Predict the labels of the new data set (with rfecv.predict(newdata)) for each of the 100 created models 
# and calculate how many times a sample is assigned to each group. Assign sample to group with the highest value
