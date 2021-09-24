from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier


# function to 
def classification_report(model_name, test, clusters):
    """
    Print a classification model report.
    """
    
    print(model_name, ":\n")
    print("Accuracy Score: ", '{:,.3f}'.format(
        float(accuracy_score(test, clusters)) * 100
    ), "%")
    print("     Precision: ", '{:,.3f}'.format(
        float(precision_score(test, clusters, average='macro')) * 100
    ), "%")
    print("        Recall: ", '{:,.3f}'.format(
        float(recall_score(test, clusters, average='macro')) * 100
    ), "%")
    print("      F1 score: ", '{:,.3f}'.format(
        float(f1_score(test, clusters, average='macro')) * 100
    ), "%")
    

def test_clustering(
    X, 
    clusters, 
    random_state, 
    test_size,
    max_iter,
):
    """
    Evaluate clustering generality.
    
    This is just one way to evaluate the clustering.  Should k-means find a 
    meaningful split in the data, a classifier could predict which cluster a 
    given instance should belong to.
    """
    # test set size of 20% of the data and the random seed 42 <3
    X_train, 
    X_test, 
    y_train, 
    y_test = train_test_split(
        X.toarray(),
        clusters, 
        test_size = test_size, 
        random_state = random_state
    )
    
    print("X_train size:", len(X_train))
    print("X_test size:", len(X_test), "\n")


    # Now let's create a Stochastic Gradient Descent classifier 

    # Precision is ratio of True Positives to True Positives + 
    # False Positives. This is the accuracy of positive predictions
    # Recall (also known as TPR) measures the ratio of True 
    # Positives to True Positives + False Negatives. It measures the
    # ratio of positive instances that are correctly detected by the 
    # classifer.
    # F1 score is the harmonic average of the precision and recall. 
    # F1 score will only be high if both precision and recall are high

    # SGD instance
    sgd_clf = SGDClassifier(
        max_iter = max_iter, 
        tol = 1e-3, 
        random_state = random_state, 
        n_jobs = -1
    )
    # train SGD
    sgd_clf.fit(X_train, y_train)

    # cross validation predictions
    sgd_pred = cross_val_predict(
        sgd_clf, 
        X_train, 
        y_train, 
        cv = 3, 
        n_jobs = -1
    )

    # print out the classification report
    classification_report(
        "Stochastic Gradient Descent Report (Training Set)", 
        y_train, 
        sgd_pred
    )


    # To test for overfitting, let's see how the model generalizes
    # over the test set

    # cross validation predictions
    sgd_pred = cross_val_predict(sgd_clf, X_test, y_test, cv=3, n_jobs = -1)

    # print out the classification report
    classification_report(
        "Stochastic Gradient Descent Report (Training Set)", y_test, sgd_pred
    )


    # Now let's see how the model can generalize across the whole dataset. 

    sgd_cv_score = cross_val_score(sgd_clf, X.toarray(), y_pred, cv = 10)
    print("Mean cv Score - SGD: {:,.3f}".format(
        float(sgd_cv_score.mean()) * 100
    ), "%")