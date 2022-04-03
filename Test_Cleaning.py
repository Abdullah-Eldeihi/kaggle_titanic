import pandas as pd
import numpy as np
from Cleaning import replace_titles

test = pd.read_csv("Datasets/test.csv")

## Get title column
title = test['Name'].str.split(",").str[1].str.split(".").str[0].str.strip()
title = title.apply(replace_titles)
test['Title'] = title

## Get the size of the family.
test['Size of Family'] = test['Parch'] + test['SibSp']

## Get the fare price per family member.
test['Fare Per Person'] = test['Fare'] / (test['Size of Family'] + 1)

## Get the maturity column
def maturity(x):
    if x < 18:
        return 'Underage'
    elif x < 30:
        return 'Young Adult'
    else:
        return 'Adult'
test['Maturity'] = test['Age'].apply(maturity)

test = test.drop(['Name', 'Cabin', 'Ticket'], axis=1)

test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
test['Fare Per Person'] = test['Fare Per Person'].fillna(test['Fare Per Person'].mean())


test.to_csv("Datasets/test_cleaned.csv",index=False)

