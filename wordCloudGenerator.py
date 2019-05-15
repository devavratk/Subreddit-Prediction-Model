"""
    File name: wordCloudGenerator
    language: Python 2.x
    Author: Devavrat Kalam
    Description: Create word cloud from posts of given subreddit
"""

# Essential libraries
import praw
import os
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np


def scrapRedditData(name):
    """
    Scrap data from Reddit API and send it to createJson method
    :param name: Name of subreddit
    :return: None
    """
    reddit = praw.Reddit(client_id='rVEEB_GO06o36Q', client_secret="RSySK5ICBc213Nq5GeOx8PlQNSU",
                         password='YoPassword_123', user_agent='whoCares',
                         username='Kiritokkkk')
    try:
        subs = reddit.subreddit(name)
        topic = subs.hot(limit=700)
        count = 0

        dict = {
            'posts': []
        }

        ignored = 0
        for topicTitle in topic:
            # We only need 200 posts, so break
            if count == 500:
                break

            # This will ignore the advertised posts
            if not topicTitle.stickied:
                dict['posts'].append(topicTitle.title)
                count += 1
            else:
                ignored += 1
        wordCloud(dict)
    except:
        print(name, ' is not a valid name for subreddit')
        newName = input('Try again. Give correct name: ')
        scrapRedditData(newName)


def wordCloud(dict):
    """
    Generate word cloud
    :param dict: Dictionary which holds posts from subreddit
    :return: None
    """

    # Detect current directory path
    currentDirectory = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()

    # Generating corpus
    wholeTxt = ' '.join(dict['posts'])

    image_mask = np.array(Image.open(os.path.join(currentDirectory, 'maskuse.png')))
    cloud = WordCloud(background_color='White', stopwords=set(STOPWORDS), mask=image_mask, contour_width=3,
                      contour_color='black', max_words=2000)
    cloud.generate(wholeTxt)

    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def main(name=None):
    """
    :param name: Takes name of subreddit requested for word cloud
    :return: None
    """
    print('Word Cloud generation initiated..')
    if not name:
        name = 'disney'
    scrapRedditData(name)


if __name__ == '__main__':
    main()
