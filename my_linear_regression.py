

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn import preprocessing
import seaborn as sb
import pandas as pd

from sklearn.metrics import confusion_matrix



if __name__ == '__main__':
    filename = 'Data/rescaled.csv'
    posts = pd.read_csv(filename)
    print(posts.head())

    y = posts.ix[:, 3].values
    posts_data = X = posts.ix[:, (0, 1, 2)]
    posts_data_name  = ['wdr','dissim','leven']
    print(y)
    # checking independence between features
    # view relationship
    sb.regplot(x='wdr', y='leven', data=posts, scatter=True)


    #sb.boxplot(x='wdr', y='leven', data=posts, palette='hls')
    plt.show()


    wdr = posts['wdr']
    dissim = posts['dissim']
    leven = posts['leven']
    # check how correlated the wdr and dissim.  Below zero is less than 50% correlation
    sb.heatmap(posts_data.corr())
    plt.show()
    spearmanr_coefficient, p_value = spearmanr(wdr, leven)
    print('Spearmnar Rank Coorelation Coefficient %0.3f' %spearmanr_coefficient)


    print(posts.isnull().sum())
    # Checking if our target is ordinary or binary
    sb.countplot(x='is_bot', data=posts, palette='hls')
    plt.show()
    print(posts.info())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=47)

    LogReg = LogisticRegression()
    LogReg.fit(X_train, y_train)

    y_pred = LogReg.predict(X_test)

    confusion_matrix = confusion_matrix(y_test, y_pred)
    print('confusion_matrix')
    print(confusion_matrix)
    sb.heatmap(confusion_matrix)
    plt.show()

    print(metrics.classification_report(y_test, y_pred))
    # x = scale(posts_data)
    # # initialize logistic regression model
    # LogReg = LogisticRegression()
    # LogReg.fit(x, y)
    # # The closer to 1 the better the fit
    # print('LogReg Score:', LogReg.score(x, y))
    #
    # # get predicted values
    # y_pred = LogReg.predict(x)
    #
    # print(metrics.classification_report(y, y_pred))


'''
Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome.

Logistic Regression Assumptions

Target variable is binary
Predictive features are interval (continuous) or categorical
Features are independent of one another
Sample size is adequate – Rule of thumb: 50 records per predictor

Uses for Logistic Regression
Stock Market Predictions
Customer conversion for sales
Continuation of support based on factors

This predicts an outcome, and it also provides a probability of that prediction being correct.

True Neg  | False Pos
False Neg | True Pos

'''