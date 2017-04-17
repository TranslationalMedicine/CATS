import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

#Variables
TEST_SIZE = 0.1
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
    X_train, X_test, y_train, y_test = sk.cross_validation.train_test_split(X, y, test_size=TEST_SIZE, random_state=0)
    return(X_train, X_test, y_train, y_test)
    
def SVC_crossvalidation(X_train,y_train, X_test, y_test):
    #This function performs recursive feature selection with cross validation in a linear SVC model. 
      #It returns the selected features and the accuracy of the model in the test data set. 
    
    # Create the RFE object and compute a cross-validated score.
    svc = sk.svm.SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = sk.feature_selection.RFECV(estimator=svc, step=1, cv=sk.model_selection.StratifiedKFold(10), scoring='accuracy')
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

# main
X,y = import_data()
X_train, X_test, y_train, y_test = split_data()
features, accuracy, rfecv = SVC_crossvalidation(X_train, y_train, X_test, y_test)
