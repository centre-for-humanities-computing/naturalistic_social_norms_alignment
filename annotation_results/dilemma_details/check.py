import json
import pandas as pd
from collections import Counter
from pprint import pprint


def get_stats(df):
    df['n_issues'] = df['label'].apply(lambda x: len(x))
    print("Issues: ")
    print(df['n_issues'].value_counts())

    issues = sum(df['label'].values.tolist(), [])
    print("\n\nIssues Distr:")

    for k,v in Counter(issues).most_common():
        print(k, v, round(v/len(df), ndigits=3))

annot_1 = pd.read_json('labeled_annot_1.json')
annot_2 = pd.read_json('labeled_annot_2.json')

get_stats(annot_1)
get_stats(annot_2)