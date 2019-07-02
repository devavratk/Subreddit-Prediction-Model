# Subreddit Prediction Model:
Developed a Subreddit category predictive model which determines category of a random subreddit with 80% accuracy.

## Table of contents:
* [Introduction](#introduction)
* [Implementation Details](#implementation-details)
* [Implementation Instructions](#implementation-instructions)

## Introduction:
  As we have enormous amount of data flowing in the social media like reddit everyday, it is important that one must be up to date about the important issues. However, understanding a post belongs to which class of subreddit is really important; as we don’t want to waste our time on unimportant posts. This problem can be solved by creating a simple subreddit category prediction model.

## Implementation Details:
### Prediction models used -
- Support Vector Machine
- Random Forest

### Filtering setting over the data -
Internet posts are often noisy, i.e. they contains emojis, filler words, slangs etc. To generate a good predictive model, I have segregated the data into following filter categories -
- Without filtering of any kind,
- After removing stop words,
- After removing stop words and stemming the words.
This is done to test the accuracies of prediction for different filter settings to decide the optimal setting for final prediction.

### The objective of the Predictive model:
- Determine subreddit class of particular post,
- Maximize prediction score high as possible


## Implementation Instructions:
Download all the .py files and place them in same folder.

Only run 'dataPreprocessing.py' file.

It will ask whether you want to build word cloud. If yes, specify the name of subreddit for word cloud(This is extra feature).
After the word cloud, console will ask for
- Number of subreddits to classify and
- Names of subreddits.

It will send the control to 'redditClassifier.py' file for implematation of actual training models and predictions.
