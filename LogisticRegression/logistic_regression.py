

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


def get_correlation(outFile, data):
    # check how correlated the attributes.  Below zero is less than 50% correlation
    plt.figure()
    sb.heatmap(data.corr(), annot=True, cmap="YlGnBu",vmax=1,vmin=-1,)
    plt.savefig(outFile, format='png')
    plt.close()
    # plt.show()

def binary_check(outFile, data):
    # Checking if our target is ordinary or binary
    plt.figure()
    sb.countplot(x='is_bot', data=data, palette='hls')
    plt.savefig(outFile, format='png')
    # plt.show()
    plt.close()
    print(data.info())

def generate_confusion_matrix(matrix,outFile,title):
    plt.figure()
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    sb.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=['non-bot', 'bot'],
               yticklabels=['non-bot', 'bot'])
    plt.savefig(outFile, format='png')
    # plt.show()
    plt.close()

def combined_embeddings(files):
    for name in files:
        embedding = name
        filename = 'Data/combined_embeddings/'+embedding+'.csv'
        posts = pd.read_csv(filename)
        # print(posts.head())
        if name == 'mds':
            y = posts.ix[:, 0].values
            posts_data = X = posts.ix[:, (1, 2)]
        else:
            y = posts.ix[:, 3].values
            posts_data = X = posts.ix[:, (0, 1, 2)]
        print(y)

        get_correlation('Results/'+embedding+'/combined/attribute_correlations.PNG', posts_data)

        print(posts.isnull().sum())
        binary_check('Results/' + embedding + '/combined/binary_target_check.PNG', posts)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=47)

        logReg = LogisticRegression()
        logReg.fit(X_train, y_train)

        y_pred = logReg.predict(X_test)

        my_confusion_matrix = confusion_matrix(y_test, y_pred)

        print('combined '+ name + 'confusion_matrix')
        print(my_confusion_matrix)
        generate_confusion_matrix(my_confusion_matrix, 'Results/'+embedding+'/combined/confusion_matrix.PNG','combined '+ name + ' confusion matrix')


        with open('Results/'+embedding+'/combined/stats.txt', 'w+') as statsFile:
            statsFile.write(metrics.classification_report(y_test, y_pred))


def separated_embeddings(files):
    for file in files:
        if file == 'mds':
            data_range = (0,1)
        else:
            data_range = (0,1,2)

        bot = 'Data/separated_embeddings/' + file + '/bot.csv'
        bot = pd.read_csv(bot)
        bot_data = bot.ix[:, data_range]
        get_correlation('Results/' + file + '/bot/attribute_correlations.PNG', bot_data)

        nonbot = 'Data/separated_embeddings/' + file + '/nonbot.csv'
        nonbot = pd.read_csv(nonbot)
        nonbot_data = nonbot.ix[:, data_range]
        get_correlation('Results/' + file + '/nonbot/attribute_correlations.PNG', nonbot_data)


if __name__ == '__main__':
    files = ['zscore', 'recoded', 'ranked', 'rescaled', 'mds']

    combined_embeddings(files)
    separated_embeddings(files)


