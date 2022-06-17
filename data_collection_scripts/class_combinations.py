"""
Class label combinations
"""
import csv
import itertools

mh_labels = ['Anxiety', 'Body Dysmorphia', 'Depression', 'Eating Disorder', 'PTSD', 'Suicide', 'none']
emotic_labels = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval',
                 'Disconnection',
                 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue',
                 'Fear', 'Happiness',
                 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']


possible_combs = list(itertools.product(mh_labels, emotic_labels))
with open('Book1.csv', 'w', newline='') as file:
    for c in possible_combs:
        print(c, sep=', ')
        writer = csv.writer(file)
        writer.writerow([c])
