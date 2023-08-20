#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:42:15 2023

@author: macintosh
"""

import pandas as pd

# Load the datasets
dataset1 = pd.read_csv('path_to_first_dataset.csv')
dataset2 = pd.read_csv('path_to_second_dataset.csv')

# Convert date columns to datetime format
dataset1['Date'] = pd.to_datetime(dataset1['Date'])
dataset2['Date'] = pd.to_datetime(dataset2['Date'])

# Merge the datasets based on the Date column
merged_dataset = pd.merge(dataset2, dataset1, on='Date', suffixes=('_dataset2', '_dataset1'))

# Columns to compare and correct
columns_to_correct = ['Open_dataset2', 'High_dataset2', 'Low_dataset2', 'Close_dataset2', 'Adj Close_dataset2', 'Volume_dataset2']
columns_correct = ['Open_dataset1', 'High_dataset1', 'Low_dataset1', 'Close_dataset1', 'Adj Close_dataset1', 'Volume_dataset1']

# Iterate through each column and correct the prices
for col_to_correct, col_correct in zip(columns_to_correct, columns_correct):
    merged_dataset[col_to_correct] = merged_dataset[col_correct]

# Drop the columns from the first dataset
merged_dataset.drop(columns_correct, axis=1, inplace=True)

# Save the corrected dataset
merged_dataset.to_csv('corrected_dataset.csv', index=False)