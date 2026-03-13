from vllm import LLM, SamplingParams
import pandas as pd
import os, json



class Prompter:
    @classmethod
    def prepare_input_text(cls, row, transcripts_dir):
        segments = json.load(
            open(
                os.path.join(transcripts_dir, row['filename']), 'r'
            )
        )["segments"]

        display_segments = [
            s for s in segments if s['id'] <= row['idx_segment_start'] + 25 and s['id'] >= row['idx_segment_start'] - 25
        ]
        display_segments = list(sorted(display_segments, key=lambda x: x['id']))

        output = f"""# Dilemma:\n{row["dilemma"]}\n\nSegment:\n"""
        for s in display_segments:
            if s['id'] == row['idx_segment_start']:
                output += ">> "
            output += f"""{s['speaker']}: {s['text']}\n"""
        output = output.strip()

        return output

    @staticmethod
    def prepare_prompt(row, transcripts_dir):
        input_data = Prompter.prepare_input_text(row, transcripts_dir)

        format_output = """{
                "reasoning": str, // reasoning on why you made this decision
                "prediction": "MATCH" | "NOT MATCH" // prediction
            }"""

        instructions_prompt = f"""You are provided with a short dilemma in Danish and a part of the transcript of the podcast (segment) in Danish.

In the podcast, hosts are introducing a dilemma by either reading a letter from a listener of the podcast, or inviting the listener to introduce the dilemma. After that, the hosts discuss on how to address the dilemma, what to do and what not to do etc.
Your task is to predict on whether the provided part of the transcript of the podcast contains *an introduction to the dilemma* or if it if right after it.

You need to predict one of the following two classes:
- MATCH
- NO MATCH

Here are some guidelines on labels:
# MATCH
- In the transcript the hosts are clearly introducing the dilemma. They provide required context and a clear introduction of the dilemma.
- Hosts are introducing the dilemma, but it is clear that in the previous sentences they have started the introduction.
- Sometimes the letter that the hosts are reading does not contain names that sre in the provided dilemma. However, if in this letter the semantics is the same, as in the dilemma, it is a match.
For example:
Dilemma: "Anna and Andrew are not sure if they can go to the bar together"
Segment: "My husband and I were living together for 3 years now. We have never drink, but we want to start. Is it ok for us to go to the pub/bar together?"
- Ignore typos and minor differences (e.g. Sara/Sarah etc.)


# NOT MATCH
- There is no indication of the introduction of the dilemma.
- The dilemma that is introduced is different from the provided one.
- Hosts are discussing the dilemma, but they are in the middle of the discussion - it is clear that the dilemma was introduced really long time ago.
- There is no indication of the introduction of the dilemma.

Basically, when deciding on the label, you need to answer the question: "Is the provided dilemma is introduced in this transcript segment?"

# Output Format
Generate an output in the following json format:
```json
{format_output}
```
Output ONLY json. Make sure that this json can be loaded via `json.loads`"""

        instructions_prompt += f"""

# Input
{input_data}"""

        return [
            {
                "role": "user",
                "content": instructions_prompt
            }
        ]



def parse_pred(raw_output):
    if "assistantfinal" in raw_output:
        raw_pred = raw_output.split("assistantfinal")[1]
    else:
        raw_pred = raw_output

    if raw_pred.startswith('```json'):
        raw_pred = raw_pred[len('```json'):]
    if raw_pred.startswith('```'):
        raw_pred = raw_pred[len('```'):]
    if raw_pred.endswith('```'):
        raw_pred = raw_pred[:-len('```')]

    try:
        prediction = json.loads(raw_pred)
        return prediction
    except:
        return {}


sampling_params = SamplingParams(max_tokens=None)

MODELNAME = "openai/gpt-oss-20b"

DF_LOADPATH = "ApproxDilemmasSegmentLocation.json"
transcripts_dir = "SpeakerDiarizedTranscripts-SaraOgMonopolet"

SAVEPATH = "GptOss20b-mismatchDilemmaPreds.json"

llm = LLM(
    model=MODELNAME,
    # max_num_batched_tokens=32768,
    gpu_memory_utilization=0.95,
    # tensor_parallel_size=4
    # pipeline_parallel_size=4,
    max_num_seqs=64
)

df = pd.read_json(DF_LOADPATH)

prompts = [
    Prompter.prepare_prompt(
        row=df.iloc[i],
        transcripts_dir=transcripts_dir
    ) for i in range(len(df))
]

outputs = llm.chat(prompts, sampling_params)
raw_extracted_solutions = [
    x.outputs[0].text for x in outputs
]

parsed_out = [parse_pred(x) for x in raw_extracted_solutions]

df[f'raw_mismatch_predictions - {MODELNAME}'] = raw_extracted_solutions
df[f'parsed_mismatch_predictions - {MODELNAME}'] = parsed_out

df.to_json(SAVEPATH)