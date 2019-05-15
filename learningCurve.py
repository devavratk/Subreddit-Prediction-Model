"""
    File name: learningCurve
    language: Python 2.x
    Author: Devavrat Kalam
    Description: Create learning curve
"""

# Essential libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(classifierObj, title, X, Y, cv=None, n_jobs=None):
    """
    Plots learning curve
    :param classifierObj: Classifier used for prediction
    :param title: Name of graph
    :param X: Features of testing set
    :param Y: True outputs
    :param cv: Cross validation iterator. Here, we are using 100 iterations
    :param n_jobs: Number of jobs which will run parallelly
    :return: None
    """

    # training examples that will be used to generate the learning curve.
    train_sizes = np.linspace(.1, 1.0, 5)

    # Figure helps separate two images so they can be plotted later differently
    plt.figure()
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy of Model")
    train_sizes, train_scores, test_scores = learning_curve(classifierObj, X, Y,
                                                            cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()


def main(name, classifier, X, Y):
    """
    :param name: name of classifier being used
    :param classifier: Classifier object
    :param X: Features of testing set
    :param Y: True output of test set
    :return: None
    """
    # Cross validation iterator with 100 iterations with 20% testing size
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    # Plot the curve
    plot_learning_curve(classifier, name, X, list(Y.target), cv=cv, n_jobs=4)


if __name__ == '__main__':
    main()
