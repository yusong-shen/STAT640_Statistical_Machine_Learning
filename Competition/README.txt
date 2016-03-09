========================================================

Data Description for Dating Profile Recommender
Stat 640 / Stat 444
Fall 2015
Genevera Allen
Rice University

=======================================================
Data Files:

-ratings.csv (A 3279759 x 3 matrix of observed ratings in the format
User ID, Profile ID, and Rating.)

-gender.csv (A 10000 x 2 matrix of user IDs and their gender - M or F.)

-IDMap.csv (A 500000 x 3 matrix where each row corresponds to one of
 the ratings you must predict; this file maps the User ID and Profile
 ID to the Kaggle ID that must be used for all Kaggle submissions.)

-README.txt  (This document!)

-PMbenchmark.R  (An example R script to read in the data, predict the
 desired ratings by their profile means, and then prepare a Kaggle
 submission file in the correct format.)

===================================================
Data Description:

For many users, the idea of sorting through hundreds of thousands of
online dating profiles to find potential matches seems daunting.
Instead, it would be great to have an automated system that recommends
profiles of other users that a user will like.  One way to accomplish
this is to build a recommendation system that predicts the profiles a
user is likely to enjoy based upon the user's past ratings of other
profiles.  To build such a recommender system, we will be working with
a small subset of profile rating data from the Czech dating site
http://libimseti.cz/.   

For this task, you are given training data consisting of 3,279,759
ratings of 10,000 profiles by 
10,000 users.  Ratings are integers between 1 and 10 with 10 being the
best.  While the training ratings can be organized as a 10,000 x
10,000 dimensional matrix with ~96.7% missing ratings, the ratings are
given to you in a compressed format.  In ratings.csv, each row is one
observed rating annotated with the user and profile IDs.  Examples of
how to convert this to a ratings matrix format are given in
PMbenchmark.R.  

Your goal is to build a recommender system for online dating profiles
based upon users' past ratings of other profiles.  Instead of
predicting all of the missing ratings, we will focus on predicting a
subset of 500,000 missing ratings so that the task (and upload time!)
is manageable.  This subset of ratings is given by their user and
profile IDs in the IDMap.csv file.  This file also maps these IDs to a
unique Kaggle ID that must be used to submit files to the
leaderboard, as described below.  

Finally, the benchmark script contains R code to produce the profile
mean benchmark as well as format the entries for submission to
the leaderboard.  

=======================================================
How to Submit an Entry to Kaggle:

Submissions to the Kaggle leaderboard must be of a specific form.
Submissions should be a matrix with two
columns, ID and Prediction and 500,000 rows corresponding to the
subset of missing ratings that you are to predict.  The "ID" is a
unique numeric Kaggle identifier from 1 to 
500,000 that corresponds to the user and profile IDs given in the
IDMap.csv file.  The "Prediction" is your predicted rating that can be
an integer or a double.   R code for the benchmark profile mean
recommender as well as code for 
formatting a submission are given in PMbenchmark.R.  

================================================================
Authors & Citation:
Lukas Brozovsky and Vaclav Petricek, "Recommender System for Online
Dating Service", In Proceedings of Conference Znalosti, 2007.
=================================================================
