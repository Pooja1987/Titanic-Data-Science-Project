import pandas as pd
import numpy as np

#getting data
test_df = pd.read_csv("C:/Users/Pooja/AppData/Local/Programs/Python/Python38-32/Junior-Data-Science-Software-Engineer-master/data/valnew.csv")
train_df = pd.read_csv("C:/Users/Pooja/AppData/Local/Programs/Python/Python38-32/Junior-Data-Science-Software-Engineer-master/data/train.csv")

train_df.info()
train_df.describe()
train_df.head(8)

#data preprocessing
train_df = train_df.drop(['PassengerId'], axis=1)

#missing Data Cabin
import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
# we can now drop the cabin feature
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)

#missing Data Age
data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)
train_df["Age"].isnull().sum()

#missing data embarked
common_value = 'S'
data = [train_df,test_df]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
	
train_df.info()
	
	
	

