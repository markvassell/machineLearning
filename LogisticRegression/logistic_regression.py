

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


def get_correlation(outfile, data):
    # check how correlated the attributes.  Below zero is less than 50% correlation
    plt.figure()
    sb.heatmap(data.corr(), annot=True, cmap="YlGnBu")
    plt.savefig(outfile, format='png')
    # plt.show()


def combined_embeddings(files):
    for name in files:
        embedding = name
        filename = 'Data/combined_embeddings/'+embedding+'.csv'
        posts = pd.read_csv(filename)
        print(posts.head())

        y = posts.ix[:, 3].values
        posts_data = X = posts.ix[:, (0, 1, 2)]
        print(y)

        get_correlation('Results/'+embedding+'/combined/attribute_correlations.PNG', posts_data)

        print(posts.isnull().sum())
        # Checking if our target is ordinary or binary
        sb.countplot(x='is_bot', data=posts, palette='hls')
        plt.savefig('Results/'+embedding+'/combined/binary_target_check.PNG', format='png')
        plt.show()
        print(posts.info())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=47)

        logReg = LogisticRegression()
        logReg.fit(X_train, y_train)

        y_pred = logReg.predict(X_test)

        my_confusion_matrix = confusion_matrix(y_test, y_pred)
        print('confusion_matrix')
        print(my_confusion_matrix)
        sb.heatmap(my_confusion_matrix,annot=True,fmt="d",cmap="YlGnBu",xticklabels=['bot','non-bot'], yticklabels=['bot','non-bot'])
        plt.savefig('Results/'+embedding+'/combined/confusion_matrix.PNG', format='png')
        plt.show()

        with open('Results/'+embedding+'/combined/stats.txt', 'w+') as statsFile:
            statsFile.write(metrics.classification_report(y_test, y_pred))


def separated_embeddings(files):
    for file in files:
        bot = 'Data/separated_embeddings/' + file + '/bot.csv'
        bot = pd.read_csv(bot)
        bot_data = bot.ix[:, (0, 1, 2)]
        get_correlation('Results/' + file + '/bot/attribute_correlations.PNG', bot_data)

        nonbot = 'Data/separated_embeddings/' + file + '/nonbot.csv'
        nonbot = pd.read_csv(nonbot)
        nonbot_data = nonbot.ix[:, (0, 1, 2)]
        get_correlation('Results/' + file + '/nonbot/attribute_correlations.PNG', nonbot_data)


if __name__ == '__main__':
    files = ['zscore', 'recoded', 'ranked', 'rescaled']

    #combined_embeddings(files)

    separated_embeddings(files)

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