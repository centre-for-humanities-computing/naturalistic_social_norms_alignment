from tqdm import tqdm
import os, json, argparse
import concurrent.futures
import pandas as pd

from openai import OpenAI
import os

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


class DataPromptPreparation:

    @staticmethod
    def prepare_section_messages(dilemma, transcript_sentence_segments, id_start, id_end, n_sections):

        before_section_segments = [
            sentence_segment_dict for sentence_segment_dict in transcript_sentence_segments if sentence_segment_dict[0]["id"] < id_start
        ][-n_sections:]

        middle_section_segments = [
            sentence_segment_dict for sentence_segment_dict in transcript_sentence_segments if
            sentence_segment_dict[0]["id"] >= id_start and sentence_segment_dict[-1]["id"] <= id_end
        ]

        after_section_segments = [
            sentence_segment_dict for sentence_segment_dict in transcript_sentence_segments if
            sentence_segment_dict[-1]["id"] > id_end
        ][:n_sections]

        section_segments = before_section_segments + middle_section_segments + after_section_segments
        section_segments = sum(section_segments, [])


        instructions_prompt = """You are a precise and reliable information extraction assistant.

You will be given:
1. A section of a Danish podcast transcript as a **context**.
2. A short version of the **dilemma**, which is needed to be extended.

In the podcast, the host will be introducing the dilemma for the group (can be referred as Monopolet) to discuss. The host will read the letter from the listener, where the dilemma with all the relevant context will be provided.

Your task is to:
- Based on the context, you need to extend the dilemma with ALL additional information.
- Do not omit any details. It is important, that you will include all the details and context that listener provided in the letter that the host reads. 
- You need to provide a question that was presented in the podcast to discuss regarding this dilemma.
- Output text in Danish!
- Do not generate any information that the listener did not provide.
- Generate an extended dilemma not as a letter, but as a text that can be presented to other person. At the end, generate a question with the question mark on what the listener asks for.


### Output Format
- Output only extended version of the dilemma with a question. Put the question at the end, and include the question mark.
- Do not output anything else - no comments, additional thoughts etc.

Your input:

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
    def prepare_prompts(parsed_approx_dilemmas_sections, sections_speakers_loaddir, n_sections):
        input_prompts = []
        prompts_meta = []


        for filename in tqdm(parsed_approx_dilemmas_sections, desc="Preparing prompts"):
            for dilemma in parsed_approx_dilemmas_sections[filename]:

                if filename in os.listdir(sections_speakers_loaddir):

                    transcript_data =  json.load(open(os.path.join(sections_speakers_loaddir, filename)))

                    transcript_sentence_segments = segments_to_sentences(transcript_data["segments"])

                    input_prompts.append(
                        DataPromptPreparation.prepare_section_messages(
                            dilemma=dilemma,
                            transcript_sentence_segments=transcript_sentence_segments,
                            id_start=parsed_approx_dilemmas_sections[filename][dilemma]["selected"]["start"],
                            id_end=parsed_approx_dilemmas_sections[filename][dilemma]["selected"]["end"],
                            n_sections=n_sections
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

    input_prompts, prompts_meta = DataPromptPreparation.prepare_prompts(
        parsed_approx_dilemmas_sections=parsed_approx_dilemmas_sections,
        sections_speakers_loaddir=args.transcript_loaddir,
        n_sections=args.cont_window
    )

    client = OpenAI(
        base_url=args.base_url if args.base_url else None
    )

    os.makedirs(args.save_dir, exist_ok=True)
    samples_save_dir = os.path.join(args.save_dir, "generated_samples")
    os.makedirs(samples_save_dir, exist_ok=True)

    modelname_save = args.modelname.replace("/", "_")
    df_savepath = os.path.join(args.save_dir, f"ExtendedDilemmas -- {modelname_save} -- ContextWindow{args.cont_window}.json")

    run_generations(
        client=client,
        conversations=input_prompts,
        savedir=samples_save_dir,
        meta_data=prompts_meta,
        modelname=args.modelname,
        max_workers=args.max_workers
    )

    print("====== Generations done ======")

    gpt_df = agg_data_into_json(samples_save_dir)
    gpt_df_expanded = pd.concat([gpt_df.drop('meta_data', axis=1), pd.json_normalize(gpt_df['meta_data'])], axis=1)

    gpt_df_expanded.to_json(df_savepath)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Extending and rephrasing the dilemmas with OpenAI"
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
        "--parsed_approx_dilemmas_sections_path",
        type=str,
        default="ApproxDilemmasSegmentLocation.json",
        help="Approximate dilemmas mapping in the transcripts",
    )

    parser.add_argument(
        "--transcript_loaddir",
        type=str,
        default="SpeakerDiarizedTranscripts-SaraOgMonopolet",
        help="path to dir, which contains transcript json files (with ot without speakers)",
    )

    parser.add_argument(
        "--cont_window",
        type=int,
        default=40,
        help="Context window to provide to LLM before and after approximate location of the dilemma. So, the model will be shown cont_window + cont_window + 1 sentences",
    )

    parser.add_argument(
        "--save_dir",
        type=int,
        default="extended_dilemmas",
        help="savedir, where to save predictions",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)


