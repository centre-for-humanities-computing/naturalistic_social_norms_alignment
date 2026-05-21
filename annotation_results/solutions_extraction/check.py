import json
import pandas as pd

labeled_data = json.load(
    open('annot_1.json', 'r')
)

labeled_df = pd.DataFrame(
    [
        {
            'id': x['id'],
            'label': x['label'],
            'n_issues': len(x['label']),
            'n_solutions': x['data'].count("DAC-") + x['data'].count("DNC-")+ x['data'].count("TAC-")+ x['data'].count("TNC-")
                         } for x in labeled_data
    ]
)

overall_labeled_df['label_final'] = overall_labeled_df['label'].apply(lambda x: x.split('.')[-1] if isinstance(x, str) else "No issue")

print("N issues: ", overall_labeled_df.n_issues.sum())
print("Ratio Issues: ", overall_labeled_df.n_issues.sum()/overall_labeled_df.n_solutions.sum())
print("N solutions: ", overall_labeled_df.n_solutions.sum())

print(overall_labeled_df['label_final'].value_counts())