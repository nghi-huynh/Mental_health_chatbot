"""
    Title: Dataset Merger for Mental Health Reddit posts
    Author: Keelin Sekerka-Bajbus B00739421
    Filename: dataset_merger.py
    Program Description:
        This program will merge multiple datasets for use in modelling mental health status detection from journal
        entries.
"""
import csv
import pandas as pd
import os
import sys
import numpy as np

new_df = pd.DataFrame()
triggers_df = pd.read_csv('../data/trigger_warnings_dataset.csv', index_col=0)
mh_df = pd.read_csv('../data/dataset_mh_journal.csv', index_col=0)
none_df = pd.read_csv('../data/dataset_none_class_entries.csv', index_col=0)

new_df = pd.concat([triggers_df, mh_df])
new_df.drop_duplicates(keep='first', inplace=True)
new_df[new_df['class'] == 'Dysmorphia'] = 'Body Dysmorphia'
new_df = new_df[new_df['class'].isin(['Suicide', 'Depression', 'PTSD', 'Anxiety', 'Eating Disorder', 'Body Dysmorphia'])]
new_df.dropna()
new_df.reset_index()
new_df = pd.concat([new_df, none_df])
new_df.dropna()
# drop other classes
print(new_df['class'].unique())
print(new_df.shape)
print(triggers_df.shape, mh_df.shape)

new_df.to_csv('dataset_merged_mh_journalling.csv', encoding='utf-8')