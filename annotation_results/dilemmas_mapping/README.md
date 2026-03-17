To evaluate the results of the mapping, 2 annotators were presented with the random one-sentence dilemmas from the metadata and a sentence chunks of the transcript. Each annotator was provided with 150 samples, with 25 samples intersection to measure an agreement. The annotators were instructed to select a binary label on whether on whether the dilemma was introduced in the provided chunks. Then, the human results were compared with the labels produced by approach. After the annotation, 240 samples were marked as matched (dilemmas introduction is in the chunks) and 60 as not matched. 

|              | Prec | Rec | F1  | Num |
|--------------|-----:|----:|----:|----:|
| Matched      | 0.99 | 0.97 | 0.98 | 240 |
| Not Matched  | 0.88 | 0.97 | 0.92 | 60  |
| **Accuracy** |  |  | **0.97** | 300 |
| **Macro**    | 0.94 | 0.97 | 0.95 | 300 |
| **Weight**   | 0.97 | 0.97 | 0.97 | 300 |

For code see [check.py](check.py)