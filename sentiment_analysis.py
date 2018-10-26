from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import numpy as np


if __name__ == '__main__':
    data = []
    labels = []

    with open('rt-polarity.neg', 'r') as file:
        for i in file: 
            data.append(i) 
            labels.append('neg')

    with open('rt-polarity.pos', 'r') as file:
        for i in file: 
            data.append(i) 
            labels.append('pos')

    del(file)

    vectorizer = CountVectorizer(ngram_range=(1,2),
                                analyzer='word',
                                encoding='latin-1')

    X = vectorizer.fit_transform(data)

    X_train, X_test, y_train, y_test  = train_test_split(
            X, 
            labels,
            train_size=0.80,
            test_size=0.2, 
            random_state=0)

    # Experiment for logistic regression#########################################

    log_model = LogisticRegression()

    penalty = ['l1', 'l2']

    C = np.linspace(0.001, 2, 10)

    hyperparameters = dict(C=C, penalty=penalty)

    clf = GridSearchCV(log_model, hyperparameters, cv=5, verbose=1, n_jobs = 1, refit='true')
    clf.fit(X_train, y_train)

    print("Best estimator for logistic regression:", clf.best_estimator_)

    y_pred = clf.best_estimator_.predict(X_test)

    print(accuracy_score(y_test, y_pred))

    ############################################################################
    
    Experiment for Naive Bayes####################################################
    nb_model = BernoulliNB()
    
    fit_prior = [True,False]

    alpha = np.linspace(0.001, 2, 20)

    hyperparameters = dict(alpha=alpha, fit_prior=fit_prior)

    clf = GridSearchCV(nb_model, hyperparameters, cv=5, verbose=1, n_jobs = 1, refit='true')
    clf.fit(X_train, y_train)

    print("Best estimator for Bernoulli Naive Bayes:", clf.best_estimator_)

    y_pred = clf.best_estimator_.predict(X_test)

    print(accuracy_score(y_test, y_pred))

    ##############################################################################

    # Experiment for Linear kernel support Vector Machine###########################
    lsvc_model = LinearSVC()

    loss = ['squared_hinge','hinge']
    
    penalty = ['l2']

    C = np.linspace(0.001, 2, 20)

    tol = np.linspace(0.00001, 0.001, 20)

    hyperparameters = dict(loss=loss, penalty=penalty, C=C, tol=tol)

    clf = GridSearchCV(lsvc_model, hyperparameters, cv=5, verbose=1, n_jobs = 1, refit='true')
    clf.fit(X_train, y_train)

    print("Best estimator for Linear kernel Support Vector Machine:", clf.best_estimator_)

    y_pred = clf.best_estimator_.predict(X_test)

    print(accuracy_score(y_test, y_pred))

    ################################################################################

    print(confusion_matrix(y_test,y_pred))
