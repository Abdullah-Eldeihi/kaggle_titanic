# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:30:16 2022

@author: Abdullah
"""


import pandas as pd
import numpy as np


# =============================================================================
# embarked
# C = Cherbourg, Q = Queenstown, S = Southampton
# =============================================================================


df = pd.read_csv("Datasets/train.csv")

# =============================================================================
# Get statistical information on the relevant numerical columns.
# Early observations on the missing values in the data.
# Check for NANs in Cabin and decide if it is worth removing.
# Impute with the mean if it is a small number of nans
# Get the title, first name and the last name in a seperate column from Name column.
# Replace the titles with only Mr/Mrs/Miss.
# Get the size of the family.
# Get the fare price per family member.
# =============================================================================

## 1: The only relevant numerical columns that will make sense to use describe on is 
## Age and Fare.
df[['Age', 'Fare']].describe()
## There are some missing data in the Age column which will be imputed later with the mean.

## 2: Checking missing data.
print(df.isnull().sum())
## Since the majority of the Cabin column is missing it will be removed.
## There are only two rows with that are missing the Embarked so we can either impute them with
## mode or remove them, it won't have that much of an effct on the data.

## 3: Dropping the Cabin column.
df = df.drop('Cabin', axis=1)

## 4: Imputing the Age colum with the mean of the column.
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Age'].isnull().sum()

## 5.1: Getting the title by itself and replacing the unnecessary titles with the common ones.
title = df['Name'].str.split(",").str[1].str.split(".").str[0].str.strip()
title.value_counts()
def replace_titles(x):
    if x in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir']:
        return 'Mr'
    elif x in ['Countess', 'Mme', 'Lady', 'the Countess']:
        return 'Mrs'
    elif x in ['Mlle', 'Ms']:
        return 'Miss'
    elif x =='Dr':
        return 'Mr'
    else:
        return x
    
title = title.apply(replace_titles)
    
title.value_counts()
df['Title'] = title

## 5.2: Get the last name.
last_name = df['Name'].str.split(",").str[0].str.strip()
df['Last Name'] = last_name

## 5.3: Get the first name.
first_name = df['Name'].str.split(",").str[1].str.split(".").str[1].str.strip()

df['First Name'] = first_name

## 6: Get the size of the family.
df['Size of Family'] = df['Parch'] + df['SibSp']

## 7: Get the fare price per family member.
df['Fare Per Person'] = df['Fare'] / (df['Size of Family'] + 1)

## 8: Replace the null values with the mode.
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Embarked'].isnull().sum()

## 9: Save the csv file.
df.to_csv("Datasets/titanic_cleaned.csv", index=False)