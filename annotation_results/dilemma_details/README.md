This repo contains annotation results for the evaluation of dilemma details.


Two Danish annotators were provided with a list of gpt-5-mini generated dilemmas and sections of the transcripts corresponding to them. They were instructed to read the generated dilemma and the discussion and mark any potential issues with the dilemma. The issues included: Missing Important Information from the transcript in the dilemma (Missing) (with regards to potential advice), Misunderstood Important Info (Mis-Un) (e.g. incorrect pronoun resolution etc), Hallucination Important Info (Hall) (completely made-up details that were not in the transcript). Additionally, if the dilemma was not in the transcript, annotators marked it as well (Not-In).
The information was considered to be important, if by subjective view of the annotator that information would change the advise that the annotator would provide in the context of the dilemma. This makes the score more subjective, but it is expected due to the nature of the task.
The annotators were provided with 150 samples each.
The authors held an annotation session with the annotators and labeled a 5 samples together to demonstrate the approach and to explain the task.

Out of 300 dilemmas, the 74 of them included some issue with it (24%). It is a large number, but at the same time the importance is a subjective concept.

| Issue Type                         | Count | Proportion |
|------------------------------------|------:|-----------:|
| Missing Important Information      | 47    | 0.157      |
| Dilemma not in provided transcript | 16    | 0.053      |
| Misunderstood Important Info       | 10    | 0.033      |
| Hallucination Important Info       | 4     | 0.013      |

For code see [check.py](check.py)