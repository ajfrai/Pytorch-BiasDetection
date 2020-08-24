#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:02:00 2020

@author: fraifeld-mba
"""

import s3fs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import output_from_notebook
import scipy.stats
import os
import uuid


def get_tweets_sample_predictions():

    tweets = pd.read_csv('s3://pytorch-hackathon-bias-detection/labeled_tweets.csv')
    
    tweets_sample = tweets.sample(frac=.1).reset_index(drop=True)
    tweets_sample["Prediction"] =          [output_from_notebook.predict(output_from_notebook.model,tweets_sample.Text[i]) for i in range(len(tweets_sample))]
    white = tweets_sample[tweets_sample.White > tweets_sample.AA].Prediction
    aa = tweets_sample[tweets_sample.AA > tweets_sample.White].Prediction
    return (white,aa)
    

def write_bias_graph(white,aa):
    white.hist(color='blue',alpha=.3)
    aa.Prediction.hist(color='orange',alpha=.3)
    handles = [Rectangle((0,0),1,1,color=c,ec="k",alpha=.2) for c in ['blue','orange']]
    labels= ["More White Words","More AA Words"]
    plt.title("Education Level Prediction Histogram For Tweets with Race-Word Distributions")
    plt.legend(handles, labels)
    plt.xlabel("Education Level")
    plt.save("BiasDetectionHistogram.png")

def write_biased_result(white,aa):
    test = scipy.stats.ttest_ind(white,aa,equal_var=False)
    if test[1] < .1:
        x = True
    else:
        x = False
    with open('biased.txt','w') as f:
        f.write(str(x))

def mark_most_recent(uid):
    with open("most-recent.txt",'w') as f:
        f.write(uid)

unique_id = str(uuid.uuid4())
os.mkdir(unique_id)
os.chdir(unique_id)
white,aa = get_tweets_sample_predictions()
write_bias_graph(white,aa)
write_biased_result(white,aa)
os.chdir("..")
mark_most_recent(unique_id)


    

