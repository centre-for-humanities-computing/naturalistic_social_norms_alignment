from tqdm import tqdm
import os, json, argparse
import concurrent.futures
import pandas as pd
from openai import OpenAI
import numpy


def segments_to_sentences(segments):
    sentence_ends = "!?."

    sentences = []

    current_sentence = []
    for segment_dict in segments:

        current_sentence.append(segment_dict)

        if any([segment_dict["text"].endswith(end_s) for end_s in sentence_ends]):
            sentences.append(current_sentence)
            current_sentence = []

    if current_sentence and current_sentence not in sentences:
        sentences.append(current_sentence)

    sentences = sorted(sentences, key=lambda x: x[0]["id"])

    return sentences


class ConclusionsPrompter:

    @staticmethod
    def get_prev_next_dilemmas(parsed_dilemmas, dilemma_end, dilemma_start, dilemma_text):
        next_closest_start_d, next_closest_start = None, -1
        prev_closest_end_d, prev_closest_end = None, 0


        for dil_text, p in parsed_dilemmas.items():

            if dil_text != dilemma_text:
                if p["selected"]["start"] > dilemma_end:
                    if next_closest_start > p["selected"]["start"] or next_closest_start == -1:
                        next_closest_start = p["selected"]["start"]
                        next_closest_start_d = dil_text

                if p["selected"]["end"] < dilemma_start:
                    if prev_closest_end < p["selected"]["end"] or prev_closest_end == 0:
                        prev_closest_end = p["selected"]["end"]
                        prev_closest_end_d = dil_text

        return prev_closest_end_d, next_closest_start_d


    @staticmethod
    def find_section_limits(split_sections_ids, id_start):

        for i in range(len(split_sections_ids)):

            if i != len(split_sections_ids) - 1:

                if id_start >= split_sections_ids[i][0]:
                    return split_sections_ids[i][0], split_sections_ids[i+1][-1]

            else:
                return split_sections_ids[i][0], -1



    @staticmethod
    def prepare_section_messages(dilemma, transcript_sentence_segments, id_start, id_end, n_sections, next_dillema_start, prev_dillema_end, split_sections_ids, tshirt_section_start):


        consider_segments = [
            sentence_segment_dict for sentence_segment_dict in transcript_sentence_segments if
            sentence_segment_dict[0]["id"] <= next_dillema_start and sentence_segment_dict[0]["id"] >= prev_dillema_end and sentence_segment_dict[0]["id"] < tshirt_section_start
        ]


        before_section_segments = [
            sentence_segment_dict for sentence_segment_dict in consider_segments if sentence_segment_dict[0]["id"] < id_start
        ][-n_sections:]

        middle_section_segments = [
            sentence_segment_dict for sentence_segment_dict in consider_segments if
            sentence_segment_dict[0]["id"] >= id_start and sentence_segment_dict[-1]["id"] <= id_end
        ]

        after_section_segments = [
            sentence_segment_dict for sentence_segment_dict in consider_segments if
            sentence_segment_dict[-1]["id"] > id_end
        ]

        section_segments = before_section_segments + middle_section_segments + after_section_segments
        section_segments = sum(section_segments, [])


        instructions_prompt = """You are a precise and reliable information extraction assistant.

You will be given:
1. A section of a Danish podcast transcript as a **context**.
2. A short version of the **dilemma**, that you need to focus your attention on.

In the podcast, the host will be introducing the dilemma for the group (can be referred as Monopolet) to discuss. The host will read the letter from the listener, where the dilemma with all the relevant context will be provided.
After that, the group is talking about potential solutions to the presented dilemma. You need to extract *all the potential solutions* that the guests of the podcats came up with. 

Your task is to:
- Based on the discussion, extract all potential solutions that the guests of the podcats came up with: both the one that they told the person to do and the ones that were considered unaccountable (not to do).
- Do not generate any information that speakers did not provide.
- Do not imagine anyh other solutions, only ones that speakers came up with.
- Do not generate any duplicates.
- For each potential solution, include all the important context to it: conditions (if any, e.g. "Do this, if this"), order of actions (e.g. "Do this after this"), include all the required entities/participants needed for this solution.
- **Important**: Each solution should be in Danish!


### Output Format
Generate output in the json format as a dictionary with `acceptable` and `not-acceptable` keys.
`acceptable` key should include solutions that the podcast guests decided the person should do. 
`not-acceptable` key should include solutions that the podcast guests decided are not acceptable to do or the person should not do.

Here is the format:
```json
{
    "acceptable": [
        "solution 1 in danish", 
        "solution 2 in danish",
        "solution 3 in danish", 
        ...
    ],
    "not-acceptable": [
        "solution 1 in danish", 
        "solution 2 in danish",
        "solution 3 in danish", 
        ...
    ]
}
```
Make sure that your generated output can be loaded via `json.loads`! Do not output anything else, only json!!


"""

        user_input = f"""**Context**"""
        for segment_dict in section_segments:
            user_input += f"""\n{segment_dict["speaker"]}: {segment_dict["text"].strip()}\n"""

        user_input += f"""\n**Dilemma**\n{dilemma.strip()}"""

        messages = [
            {"role": "user", "content": instructions_prompt + user_input},
        ]

        return messages


    @staticmethod
    def prepare_prompts(parsed_approx_dilemmas_sections, sections_speakers_loaddir, n_sections, tshirt_section_found):
        input_prompts = []
        prompts_meta = []


      #  for filename in tqdm(list(parsed_approx_dilemmas_sections.keys())[:2], desc="Preparing prompts"):

        for filename in tqdm(parsed_approx_dilemmas_sections, desc="Preparing prompts"):
            for dilemma in parsed_approx_dilemmas_sections[filename]:

                if filename in os.listdir(sections_speakers_loaddir):

                    transcript_data =  json.load(open(os.path.join(sections_speakers_loaddir, filename)))

                    transcript_sentence_segments = segments_to_sentences(transcript_data["segments"])


                    prev_closest_end_d, next_closest_start_d = ConclusionsPrompter.get_prev_next_dilemmas(
                        parsed_dilemmas=parsed_approx_dilemmas_sections[filename],
                        dilemma_end=parsed_approx_dilemmas_sections[filename][dilemma]["selected"]["end"],
                        dilemma_start=parsed_approx_dilemmas_sections[filename][dilemma]["selected"]["start"],
                        dilemma_text=dilemma
                    )
                    if prev_closest_end_d == None:
                        prev_dillema_end = 0
                    else:
                        prev_dillema_end = parsed_approx_dilemmas_sections[filename][prev_closest_end_d]["selected"]["end"]

                    if next_closest_start_d == None:
                        next_dillema_start = max([sentence_segment_dict[0]["id"] for sentence_segment_dict in transcript_sentence_segments]) + 1
                    else:
                        next_dillema_start = parsed_approx_dilemmas_sections[filename][next_closest_start_d]["selected"]["start"]


                    input_prompts.append(
                        ConclusionsPrompter.prepare_section_messages(
                            dilemma=dilemma,
                            transcript_sentence_segments=transcript_sentence_segments,
                            id_start=parsed_approx_dilemmas_sections[filename][dilemma]["selected"]["start"],
                            id_end=parsed_approx_dilemmas_sections[filename][dilemma]["selected"]["end"],
                            next_dillema_start=next_dillema_start,
                            prev_dillema_end=prev_dillema_end,
                            n_sections=n_sections,
                            split_sections_ids=transcript_data["split_sections_ids"],
                            tshirt_section_start=tshirt_section_found[filename]["id"]
                        )
                    )
                    prompts_meta.append(
                        {
                            "filename": filename,
                            "dilemma": dilemma,
                        }
                    )

        return input_prompts, prompts_meta


def parse_text_outputs(outputs):
    return [
        output.outputs[0].text for output in outputs
    ]


def parse_output(output: str):
    if output.startswith("```json"):
        output = output[len("```json"):]
    if output.startswith("```"):
        output = output[len("```"):]
    if output.endswith("```"):
        output = output[:-len("```")]

    try:
        parsed_out = json.loads(output)
        return parsed_out["acceptable"], parsed_out["not-acceptable"]
    except:
        return [], []


def generate_save_output(client, idx, modelname, messages, save_dir, meta_data):
    if str(idx) + ".json" in os.listdir(save_dir):
        return "", idx


    output = client.chat.completions.create(
        model=modelname,
        messages=messages,
        stream=False,
        seed=2
    )

    pred = output.choices[0].message.content

    output = {
        "meta_data": meta_data,
        "output": pred
    }

    try:
        acceptable, not_acceptable = parse_output(pred)
    except:
        acceptable = []
        not_acceptable = []

    output["acceptable"] = acceptable
    output["not_acceptable"] = not_acceptable

    with open(os.path.join(save_dir, str(idx) + ".json"), 'w') as f:
        json.dump(output, f)

    return pred, idx

def run_generations(client, conversations, meta_data, modelname, savedir, max_workers):

    output_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:


        future_to_entry = {
            executor.submit(generate_save_output,
                            client, i, modelname, messages, savedir, meta_data[i]): i
            for i, messages in enumerate(conversations)
        }



        # execute whatever is in queue
        for future in tqdm(concurrent.futures.as_completed(future_to_entry), total=len(future_to_entry)):
            entry = future_to_entry[future]
            try:
                result = future.result()
                if result:
                    output_list.append(result)
            except Exception as e:
                print(f"Error {type(e)}:{e} processing entry: {entry}")

    return output_list

def agg_data_into_json(dirpath):

    data = []

    for filename in os.listdir(dirpath):
        if filename.endswith(".json"):
            sample_output = json.load(
                open(
                    os.path.join(dirpath, filename), "r"
                )
            )
            data.append(sample_output)

    return pd.DataFrame(data)



def run(args):

    parsed_approx_dilemmas_sections = json.load(open(args.parsed_approx_dilemmas_sections_path, "r"))
    tshirt_section_found = json.load(open(args.parsed_tshirt_sections_path, "r"))


    input_prompts, prompts_meta = ConclusionsPrompter.prepare_prompts(
        parsed_approx_dilemmas_sections=parsed_approx_dilemmas_sections,
        sections_speakers_loaddir=args.transcript_loaddir,
        n_sections=args.left_cont_window,
        tshirt_section_found=tshirt_section_found
    )

    if args.encoding_name:
        import tiktoken
        encoding = tiktoken.get_encoding("o200k_base")

        def get_encodings(prompt):
            return len(encoding.encode(prompt[0]["content"]))

        num_tokens = [get_encodings(s) for s in input_prompts]
        print(
            "Total tokens: ", sum(num_tokens)
        )
        print(
            "Avg: ", numpy.mean(num_tokens)
        )
        print("Std: ", numpy.std(num_tokens))
        print("Max-Min: ", numpy.max(num_tokens), numpy.min(num_tokens))


    client = OpenAI(base_url=args.base_url)

    os.makedirs(args.save_dir, exist_ok=True)

    samples_savedir = os.path.join(args.save_dir, "generated_samples/")
    os.makedirs(samples_savedir, exist_ok=True)

    df_savepath = os.path.join(args.save_dir, "agg_df.json")

    run_generations(
        client=client,
        conversations=input_prompts,
        savedir=samples_savedir,
        meta_data=prompts_meta,
        modelname=args.modelname,
        max_workers=args.max_workers
    )

    print("====== Generations done ======")

    gpt_df = agg_data_into_json(samples_savedir)
    gpt_df_expanded = pd.concat([gpt_df.drop('meta_data', axis=1), pd.json_normalize(gpt_df['meta_data'])], axis=1)

    gpt_df_expanded.to_json(df_savepath)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generating solutions suggested or advised against by the podcast guests via openai"
    )

    parser.add_argument(
        "--base_url",
        type=str,
        default="",
        help="Base url for openai api client.",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Number of worker processes to use when sending requests to OpenAI.",
    )

    parser.add_argument(
        "--modelname",
        type=str,
        default="gpt-5-mini",
        help="model to use",
    )

    parser.add_argument(
        "--encoding_name",
        type=str,
        default="",
        required=False,
        help="encoding name to load to estimate amount of input tokens. If Empty - skip this step.",
    )

    parser.add_argument(
        "--parsed_approx_dilemmas_sections_path",
        type=str,
        default="ApproxDilemmasSegmentLocation.json",
        help="Approximate dilemmas mapping in the transcripts",
    )

    parser.add_argument(
        "--parsed_tshirt_sections_path",
        type=str,
        default="TshirtSectionSegmentsLocation.json",
        help="path to json with the tshirt section mapping",
    )

    parser.add_argument(
        "--transcript_loaddir",
        type=str,
        default="SpeakerDiarizedTranscripts-SaraOgMonopolet",
        help="path to dir, which contains transcript json files (with speakers)",
    )

    parser.add_argument(
        "--left_cont_window",
        type=int,
        default=20,
        help="Context window to provide to LLM before approximate location of the dilemma.",
    )

    parser.add_argument(
        "--save_dir",
        type=int,
        default="extracted_conclusions",
        help="savedir, where to save predictions",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())