import pandas as pd
# from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class style:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def add_is_bot_col(is_bot, data):
    if is_bot == 0:
        data['is_bot'] = 0
    else:
        data['is_bot'] = 1

    return data

def combine_data(bots, non_bots):
    bots = add_is_bot_col(1, bots)
    non_bots = add_is_bot_col(0, non_bots)

    return bots.append(non_bots)


def prepare_data():
    all_data_set = list()
    ranked_bot_data_url = 'https://raw.githubusercontent.com/tonikazic/munlp_f17/master/s18/results/embeddings/mark/ranked_bot_scoringv2.csv?token=AI7HgZ-H4AARV4S8GS7AFB2jzf6N6dr-ks5a8SwiwA%3D%3D'
    ranked_non_bot_data_url = 'https://raw.githubusercontent.com/tonikazic/munlp_f17/master/s18/results/embeddings/mark/ranked_nonbot_scoring.csv?token=AI7HgdOvYt0Qn8utV2Mq-kKfa9fb6O4Mks5a8TBuwA%3D%3D'
    ranked_bot_data = pd.read_csv(ranked_bot_data_url)
    ranked_non_bot_data = pd.read_csv(ranked_non_bot_data_url)
    ranked_combined_data = combine_data(ranked_bot_data, ranked_non_bot_data)
    all_data_set.append(['Mark\'s Ranked',ranked_combined_data])

    rescaled_combined_data_url = 'https://raw.githubusercontent.com/tonikazic/munlp_f17/master/s18/results/embeddings/samaikya/rescaled_bot_nonbot/rescaled_bot_nonbot_Scoring.csv?token=AI7HgT70wLgEugGgHhYofm3KDo2uxqc5ks5a8dJ3wA%3D%3D'
    rescaled_combined_data = pd.read_csv(rescaled_combined_data_url)
    all_data_set.append(['Samaikya\'s Rescaled', rescaled_combined_data])

    scoring_combined_data_url = 'https://raw.githubusercontent.com/tonikazic/munlp_f17/master/s18/results/embeddings/derek/scoring_updated.csv?token=AI7HgVF1QPH42oTHwx2XiSv0_xtRs-KGks5a8dkEwA%3D%3D'
    scoring_combined_data = pd.read_csv(scoring_combined_data_url)
    all_data_set.append(['Derek\'s Scoring', scoring_combined_data])


    zscore_bot_data_url = 'https://raw.githubusercontent.com/tonikazic/munlp_f17/master/s18/results/embeddings/rui/bot_scoringv2.csv?token=AI7HgWrgkLiRu4rzBSHvO12kL0mmavynks5a8dtdwA%3D%3D'
    zscore_non_bot_data_url = 'https://raw.githubusercontent.com/tonikazic/munlp_f17/master/s18/results/embeddings/rui/nonbot_scoring.csv?token=AI7HgSnqRnMAKK5bRnAr5peOfnunsUe_ks5a8dukwA%3D%3D'
    zscore_bot_data = pd.read_csv(zscore_bot_data_url)
    zscore_non_bot_data = pd.read_csv(zscore_non_bot_data_url)
    zscore_combined_data = combine_data(zscore_bot_data, zscore_non_bot_data)
    all_data_set.append(['Rui\'s Zscore', zscore_combined_data])


    recoded_combined_data_url = 'https://raw.githubusercontent.com/tonikazic/munlp_f17/master/s18/results/embeddings/sai_new/new_recoded_ml.csv?token=AI7HgVaiyFLa3KzDgyctrA6zXj1CpVYeks5a8eNPwA%3D%3D'
    recoded_combined_data = pd.read_csv(recoded_combined_data_url)
    all_data_set.append(['Sai\'s Recoded', recoded_combined_data])

    return all_data_set

def main():


    data_sets = prepare_data()

    for id, data in data_sets:
        print(style.RED + style.BOLD + style.UNDERLINE + id + ' Data:' +style.END)

        # print(scatter_matrix(combined_data))
        # plt.show()

        # Create a Validation Data set
        # split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as
        # a validation data set
        validation_size = 0.15
        X = data.ix[:, (1,2,3)].values
        Y = data.ix[:, 4].values
        seed = 34
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                        random_state=seed)


        model_dict = {'LR' : 'Logistic Regression',
                      'LDA': 'Linear Discriminant Analysis',
                      'KNN':'K Neighbors Classifier',
                      'CART':'Decision Tree Classifier',
                      'NB':'Gaussian Naive Bayes',
                      'SVM':'Support Vector Classification'
                      }
        scoring = 'accuracy'
        # Spot Check Algorithms
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        # evaluate each model in turn
        results = []
        names = []
        best_model_score, best_model, model_name  = 0, None, None

        display_msg = ''
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            display_msg += "%s  \tmean = %f \tstd = (%f) \n" % (name, cv_results.mean(), cv_results.std())

            if cv_results.mean() > best_model_score:
                best_model = model
                best_model_score = cv_results.mean()
                model_name = name


        print(display_msg)

        # Make predictions on validation dataset
        selected_model = best_model
        selected_model.fit(X_train, Y_train)
        predictions = selected_model.predict(X_validation)
        print('The best model: '+ style.GREEN + model_dict[model_name] + style.END)
        print('Accuracy: ', accuracy_score(Y_validation, predictions))
        print(confusion_matrix(Y_validation, predictions))
        print(classification_report(Y_validation, predictions))

if __name__ == '__main__':
    main()


# K Neighbors Classifier
# implements learning based on the k nearest neighbors of each query point, where k is an integer value specified by
# the user

# Linear Discriminant Analysis
# A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using
# Bayes’ rule.
# The model fits a Gaussian density to each class, assuming that all classes share the same covariance matrix.
# The fitted model can also be used to reduce the dimensionality of the input by projecting it to the most
# discriminative directions.

# C-Support Vector Classification.
# The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which
# makes it hard to scale to dataset with more than a couple of 10000 samples.
