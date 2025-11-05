from tqdm import tqdm
import os, json, argparse
import pandas as pd
import concurrent.futures
from openai import OpenAI


class Prompter:

    @staticmethod
    def prepare_prompt(text, host):

        prompt = f"""You are provided with the text that was rewritten from the letter to "{host} og Monopolet" podcast in Danish. 
Your task is to remove from this text all mentions of this being a letter, as well as all references to the podcast, monopolet (guests of the podcast), host etc.

You need to remove: 
- specific markings of the letter
- rewrite questions if they addressed to the Monopolet or the host to more general ones (e.g. "What Monopolet thinks I should do?" to "What should I do ?")
- Rewrite strangely written text (like "I, who call myself Adam,...") to a natural phrasing ("My name is Adam and....")

Do not remove or omit any important details from the text. At the end it should contain a question regarding the advice. 
Your generated text should be in Danish!

Output ONLY rewritten text, nothing else. No markdown, no comments, no suggestions - only rewritten text in Danish.

# Input text:
{text}"""

        return [
            {"role": "user", "content": prompt}
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

def agg_data_into_list(dirpath, conversations):

    data = [None for _ in range(len(conversations))]

    for filename in os.listdir(dirpath):
        if filename.endswith(".json"):
            sample_output = json.load(
                open(
                    os.path.join(dirpath, filename), "r"
                )
            )
            data[sample_output['meta_data']['idx']] = sample_output['output']

    return data

def run(args):
    client = OpenAI(
        base_url=args.base_url
    )

    df = pd.read_json(args.solutions_dilemmas_all_path)

    colnames = [
        c for c in df.columns if c.startswith("first_person_extended_dilemma_") or c.startswith("extended_dilemma_")
    ]
    print(colnames)

    for c in colnames:
        print(">>>> Processing ", c)

        input_prompts = [
            Prompter.prepare_prompt(
                text=df.iloc[i][c],
                host="Mads" if "mads" in df.iloc[i]['episode_filename'] else "Sara"
            ) for i in range(len(df))
        ]

        prompts_meta = [{"idx": i } for i in range(len(df))]

        samples_savedir = os.path.join(args.save_dir, f"samples-{c.replace("/", "_")}")
        os.makedirs(samples_savedir, exist_ok=True)

        run_generations(
            client=client,
            conversations=input_prompts,
            savedir=samples_savedir,
            meta_data=prompts_meta,
            modelname=args.modelname,
            max_workers=args.max_workers
        )

        print(f"====== {c} Generations done ======")

        preds_list = agg_data_into_list(samples_savedir, input_prompts)
        df[f"noletter_{c}"] = preds_list

        df.to_json(
            os.path.join(args.save_dir, "noletter_solutions_dilemmas_all.json"),
        )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Postprocessing/rewrite of the dilemmas with removal of monopolet and letter mentions"
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
        "--solutions_dilemmas_all_path",
        type=str,
        default="first_person_rewriting_output/solutions_dilemmas_fper.json",
        help="path to json with all aggregated data in it with first and third person",
    )

    parser.add_argument(
        "--save_dir",
        type=int,
        default="noletters_processed",
        help="savedir, where to save predictions",
    )

    return parser.parse_args()

if __name__ == "__main__":
    run(parse_args())
