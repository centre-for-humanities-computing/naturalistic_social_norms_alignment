import json
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

annot_1_data = json.load(open("annot_1.json", 'r'))
annot_2_data = json.load(open("annot_2.json", 'r'))

print(classification_report(
    y_true=[x['label'][0].upper() for x in annot_1_data +  annot_2_data],
    y_pred = [x['pred']['prediction'].upper().replace("NO MATCH", "NOT MATCH") for x in pred_1_data+pred_2_data],
))

x, y = [], []


def get_intersection(sample, annot_data):
    for a in annot_data:
        if a['data'] == sample['data']:
            return a


for sample in annot_1_data:
    intersect_sample = get_intersection(sample, annot_2_data)
    if intersect_sample:
        x.append(sample['label'])
        y.append(intersect_sample['label'])


print(cohen_kappa_score(
    x,y
))