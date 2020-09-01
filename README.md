# Titanic-Data-Science-Project
# Predicting Titanic survivors 

A machine learning web application that uses historical passenger data to predict
whether someone would have survived the sinking of the Titanic.

## How it Works

The goal of the application is to predict whether a hypothetical passenger
(possibly based on the user's personal details) would have survived the sinking
of the Titanic. It is composed of a training module that learns the relation
between various passenger details and their survival outcome, and a web application
that uses the trained model to make a prediction on new passenger information.

All the modules are written in Python.

## Training

The script `train.py` processes the data, trains a few models and picks the
best one to generate the prediction for the target set and predict the results on a given dataset. This is mainly used to generate
the best score I could achieve was **0.811..**.

There are plenty of resources on how to organize a machine learning project for
this particular problem, so I won't go into too many details. I did try to figure
things out on my own as much as possible. Below is a summary of the main techniques
I used.

### Data Processing
Started with the data exploration where I got a feeling for the dataset, checked about missing data and learned which features are important
During the data preprocessing part, I computed missing values, converted features into numeric ones, grouped values into categories and created a few new features.

The features I picked for the model are:
1.A cabin number looks like ‘C123’ and the letter refers to the deck. Therefore we’re going to extract these and create a new feature, that contains a persons deck. Afterwords we will convert the feature into a numeric variable. The missing values will be converted to zero.
2.Dropped ‘PassengerId’ from the train set, because it does not contribute to a persons survival probability
3.Since the Embarked feature has only 1 missing values, we will just fill these with the most common one.
4.Converting “Fare” from float to int64, using the “astype()” function
5.converting some features into numeric

## Model Selection

The script goes through four of the models that come with Sklearn, using default
parameters. The training set is split into a training set
and a cross-validation set, and the models are compared based on the score on the CV set.

The best model according to this comparison is Random Forest classifier that goes on the first place
 with a training score of 0.81.
 Our model has a average accuracy of 81% with a standard deviation of 5 %. The standard deviation shows us, how precise the estimates are .
 Random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction

