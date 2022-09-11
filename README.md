# Stock-Price-Forecasting-based-on-Sentiment-Analysis-using-Transforme

This repo contains the following modules:

## data_generator.py

This file generates training data by reading the labeled Apple-Twitter-Sentiment.csv file.

## train3.py and train4.py

Training a classifier based on the output of Bert feature vector, the 4-th implementation is faster.

## data_cleaner.py

Provide testing data from crawled twitter csv files.

## plot_train_log.py

Visualize the training loss and precision.

## vanilla_classifier.py

The pytorch implementation of classifiers.

## test_tweets.py

Test the crawled tweets, and give each of them a sentiment score. Finally, it outputs an average score of each day's data.

## corr_calc.py

Calculate the correlation between stock and sentiment score, then visualize it.

## draw_tweet_by_date.py

Draw the tweets histogram by date.

## svm.py

The svm implementation.
