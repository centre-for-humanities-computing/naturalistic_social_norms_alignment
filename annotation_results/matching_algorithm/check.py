import json

annot_1 = json.load(
    open("annot_1.json", 'r')
)
annot_2 = json.load(
    open("annot_2.json", 'r')
)

# pairs are split by '=========='
total_pairs = sum(
    [x['data'].count('==========') for x in annot_1 + annot_2]
)
n_issues = 25 + 21

error_rate = n_issues / total_pairs
print(error_rate)