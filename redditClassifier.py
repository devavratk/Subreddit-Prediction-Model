"""
    File name: redditClassifier
    language: Python 2.x
    Author: Devavrat Kalam
    Description: Perform prediction over scrapped subreddit files
"""

# Modules to deal with files
import fnmatch
import os
# Essentials
import pandas as pd
import numpy as np
# Libraries related to Bag of words
from nltk import word_tokenize
import re
from nltk.corpus import stopwords
from autocorrect import spell
from nltk.stem.porter import PorterStemmer
# T-test
from scipy.stats import ttest_ind
# ANOVA test
import scipy.stats as stats
# Plotting graphs
import matplotlib.pyplot as plt
# Model related
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# Importing other linked python files
import learningCurve


def getFeatures(dataFiles):
    """
    Extract features from raw datafiles of subreddit posts
    :param dataFiles: list of subreddit post files
    :return: List of words extracted from files and arranged in form of filters
    """
    pt = PorterStemmer()
    stopwordsEnglish = stopwords.words('english')
    withStop = []
    withoutStop = []
    withoutSStem = []

    for filename in dataFiles:
        for i in range(filename.shape[0]):
            #         Removing stop words
            corpus = filename.iloc[i, 0]
            #         This will only contain alphabets in lowercase and remove everything else
            corpus = re.sub('[^A-Za-z]', ' ', corpus).lower()
            #         Tokenized text
            tokenizedT = word_tokenize(corpus)

            """
            We are storing words in three ways here(Filtering words).
            1) Without any filtering, i.e temp1
            2) With removing stop words, i.e temp2
            3) With removing stop words and stemming, i.e temp3
            """
            temp1 = []
            temp2 = []
            temp3 = []
            for word in tokenizedT:
                temp1.append(word)

                # Checking for stop words
                if word not in stopwordsEnglish:
                    temp2.append(word)

                    # Autocorrecting words with stemming
                    word = pt.stem(spell(word))
                    temp3.append(word)

            # Appending the collected word sets in different arrays
            withStop.append(' '.join(temp1))
            withoutStop.append(' '.join(temp2))
            withoutSStem.append(' '.join(temp3))

    wordLists = []
    wordLists.extend([withStop, withoutStop, withoutSStem])
    print("Feature extraction from Corpus complete.")
    return wordLists


def getDataframes(dataFiles, wordLists):
    """
    Creates dataframes for each wordList
    :param dataFiles: list of subreddit post files
    :param wordLists: List of words extracted from files and arranged in form of filters
    :return:
    """
    vectorizer = CountVectorizer()
    all_data_frames = []

    for filteredSent in wordLists:
        # Creating features
        X = vectorizer.fit_transform(filteredSent).toarray()

        # Creating data frame with features
        data = pd.DataFrame(X, columns=vectorizer.get_feature_names())

        # Creating labeled column for dataframe
        tempTarget = np.zeros(len(X))
        start = 0
        # Assigning appropriate labels to target data
        for i in range(len(dataFiles)):
            tempTarget[start:start + len(dataFiles[i])] = i + 1
            start += len(dataFiles[i])
        # Appending target data to the dataframe
        data['target'] = tempTarget
        all_data_frames.append(data)
    print("DataFrame creation complete.")
    return all_data_frames


def train(all_data_frames):
    """
    :param all_data_frames: File which contains all data frames for different filters
    :return: None
    """
    print(all_data_frames[0].shape)
    print(all_data_frames[1].shape)
    print(all_data_frames[2].shape)
    print()
    names = ['No Filters', 'Stopwords', 'Stopwords with Stemming']
    for data, t in zip(all_data_frames, names):
        print('Prediction on filter type - {}'.format(t))
        classifiers(data)
        print('---------------------------------------------------')

    print('We have already seen, after running above procedure generating 150 accuracy samples, \n'
          'Stopwords with Stemming filer gives high promise of accuracy. Thus we will run our '
          'final testing set on this filter set')
    print()
    print('Prediction on filter type - {}'.format('Stopwords with Stemming'))
    classifiers(all_data_frames[2], final=True)


def classifiers(data, final=False):
    """
    This method will perform splitting, fitting, training and report generation steps
    :param data: Data on which prediction has to be performed
    :param final: Whether this is final classification of the program
    :return: None
    """

    # Creating models
    clf1 = SVC(kernel='linear', probability=True)
    clf2 = RandomForestClassifier()

    train, test = train_test_split(data, test_size=0.5)

    # Splitting the test data into dev and test subsets
    dev, test = np.array_split(test, 2)

    y_train = train.loc[:, ['target']]
    y_dev = dev.loc[:, ['target']]
    y_test = test.loc[:, ['target']]

    colNames = [name for name in list(data.columns) if name != 'target']
    x_train = train.loc[:, colNames]
    x_dev = dev.loc[:, colNames]
    x_test = test.loc[:, colNames]


    for classifier, name in zip([clf1, clf2], ['SVM', 'Random Forest']):
        model = classifier.fit(x_train, y_train.values.ravel())
        if not final:
            y_predict = model.predict(x_dev)

            # Reports
            print('On {} classifier -'.format(name))
            calc_accuracyScore(y_dev, y_predict)
            calc_confusionM(y_dev, y_predict)
            calc_classificationR(y_dev, y_predict)
        else:
            y_predict = model.predict(x_test)

            # Reports
            print('Final report ', end='')
            print('on {} classifier -'.format(name))

            print('Learning curve -')
            learningCurve.main(name, classifier, x_test, y_test)

            calc_accuracyScore(y_test, y_predict)
            calc_confusionM(y_test, y_predict)
            print()
            calc_classificationR(y_test, y_predict)


def calc_accuracyScore(y, y_predict):
    print("Accuracy = {}".format(accuracy_score(y, y_predict)))


def calc_confusionM(y, y_predict):
    print("confusion_matrix =\n{}".format(confusion_matrix(y, y_predict)))


def calc_classificationR(y, y_predict):
    print("classification Report =\n{}".format(classification_report(y, y_predict)))


def main():
    dataFilesPath = os.path.join(os.path.dirname(__file__), 'datafiles')
    filenames = [f for f in os.listdir(dataFilesPath) if fnmatch.fnmatch(f, '*json')]
    print(filenames)
    dataFiles = []
    for index in range(len(filenames)):
        dataFiles.append(pd.read_json(os.path.join(dataFilesPath, filenames[index])))
    wordLists = getFeatures(dataFiles)
    all_data_frames = getDataframes(dataFiles, wordLists)
    train(all_data_frames)


if __name__ == '__main__':
    main()

