"""
    File name: dataPreprocessing
    language: Python 2.x
    Author: Devavrat Kalam
    Description: Preprocessing reddit data, i.e. making Json files
"""

# Essentials
import praw
import json
import os
import shutil
# Importing other linked python files
import wordCloudGenerator
import redditClassifier


def scrapRedditData(name):
    """
    Scrap data from Reddit API and send it to createJson method
    :param name: Name of subreddit
    :return: None
    """
    try:
        subs = reddit.subreddit(name)
        topic = subs.hot(limit=550)
        count = 0

        dict = {
            'posts': []
        }

        ignored = 0
        for topicTitle in topic:
            # We only need 200 posts, so break
            if count == 300:
                break

            # This will ignore the advertised posts
            if not topicTitle.stickied:
                dict['posts'].append(topicTitle.title)
                count += 1
            else:
                ignored += 1

        createJson(name, dict)
    except Exception as e:
        print(e)
        print(name, 'is not a valid name for subreddit')
    return


def createJson(name, file):
    """
    Create json files
    :param name: Name of subreddit
    :param file: Dictionary containing 'posts' of subreddit
    :return: None
    """
    filename = name + '.json'
    filepath = os.path.join(os.path.dirname(__file__), 'datafiles')
    print(filepath)
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    with open(os.path.join(filepath, filename), 'w') as op:
        op.write(json.dumps(file))

    # Testing
    with open(os.path.join(filepath, filename), 'r') as input:
        data = json.load(input)
        print(data)
        print(type(data))
        print(len(data['posts']))


def main():

    answer = input('Do you want to see a WordCloud before we start?\nyes / no\n')
    #
    if answer == 'yes':
        wordCloudGenerator.main(input('Type a subreddit name\n'))
    #
    print('Lets start with our predictive models.')
    names = []
    userD = int(input('Number of subreddits needed: '))
    print('Type subreddit names -\n')
    for index in range(userD):
        names.append(input())

    print('Ok, reddit scrapping initiated...')

    # names = ['jokes', 'beer', 'cannabis', 'chocolate', 'disney', 'funny']

    # Create json files
    for name in names:
        scrapRedditData(name)

    redditClassifier.main()

    # Delete the files after the program ends
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'datafiles'))


if __name__ == '__main__':
    """
        Can I have the below code here?
    """
    reddit = praw.Reddit(client_id='rVEEB_GO06o36Q', client_secret="RSySK5ICBc213Nq5GeOx8PlQNSU",
                         password='YoPassword_123', user_agent='whoCares',
                         username='Kiritokkkk')
    main()