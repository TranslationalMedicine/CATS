import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

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
    X_train, X_test, y_train, y_test = sk.cross_validation.train_test_split(X, y, test_size=TEST_SIZE)
    return(X_train, X_test, y_train, y_test)
    
def recursive_feature_elimination_cv(X_train,y_train, X_test, y_test):
    #This function performs recursive feature selection with cross validation in a linear SVC model. 
      #It returns the selected features and the accuracy of the model in the test data set. 
    
    # Create the RFE object and compute a cross-validated score.
    svc = sk.svm.SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = sk.feature_selection.RFECV(estimator=svc, step=1, cv=sk.model_selection.StratifiedKFold(3), scoring='accuracy')
    rfecv.fit(X_train, y_train)
    print("Optimal number of features : %d" % rfecv.n_features_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    ## Get the selected features 
    #(now we will get the indices of the selected features)
    features=rfecv.get_support(indices=True)
    #This will give the ranking of the features
    RankFeatures=rfecv.ranking_
    
    #Determine the accuracy of the SVC model on the test-data 
    accuracy=rfecv.score(X_test, y_test)
    
    return(features, accuracy, rfecv)

def random_forest(X_train,y_train, X_test, y_test):
    clf = sk.pipeline.Pipeline([
            ('feature_selection', sk.feature_selection.SelectFromModel(sk.svm.SVC(kernel="linear"))),
            ('classification', sk.ensemble.RandomForestClassifier())])
    accuracy = clf.fit(X_train, y_train).score(X_test, y_test)
    
    return clf, accuracy

def accuracy(rfecv, rf):
    accuracy_list = [rfecv, rf]
    names = ['Recursive Feature Elimination', 'Random Forest']
    for i in range(len(accuracy_list)):
        print('The average accuracy for %s is: %f' % (names[i], sum(accuracy_list[i])/len(accuracy_list[i])))

# main
X,y = import_data()
result_rfecv = {}
result_rf = {}
accuracy_list_rfecv =[]
accuracy_list_rf =[]
for i in range(100):
    X_train, X_test, y_train, y_test = split_data()
    features_rfecv, accuracy_rfecv, rfecv = recursive_feature_elimination_cv(X_train, y_train, X_test, y_test)
    rf, accuracy_rf = random_forest(X_train,y_train, X_test, y_test)
    result_rfecv[i] = [features_rfecv,accuracy_rfecv,rfecv]
    result_rf[i] = [accuracy_rf, rf]
    accuracy_list_rfecv.append(accuracy_rfecv)
    accuracy_list_rf.append(accuracy_rf)
accuracy(accuracy_list_rfecv, accuracy_list_rf)
