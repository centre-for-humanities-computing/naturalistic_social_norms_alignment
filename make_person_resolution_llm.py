from tqdm import tqdm
import os, json, argparse
import concurrent.futures
import pandas as pd
from openai import OpenAI


class PersonaResolutionPrompter:

    @staticmethod
    def prepare_section_messages(dilemma):

        prompt = f"""You are presented with text (usually a letter) in Danish. Your task is to rewrite it by changing all references to the main character/author from the third person to the first person.
        
For example:
Adam skriver, at hans kæreste Robin arbejder for meget, og at han savner at bruge tid sammen med hende. Hun arbejder ofte til midnat, og det påvirker virkelig deres forhold. Adam er bekymret for, at de på et tidspunkt vil slå op. Hvad kan du foreslå, at Adam gør i denne situation?

Your output should be:
Min kæreste Robin arbejder for meget, og jeg savner at bruge tid sammen med hende. Hun arbejder ofte til midnat, og det påvirker virkelig vores forhold. Jeg er bekymret for, at vi på et tidspunkt vil slå op. Hvad kan du foreslå, at jeg gør i denne situation?

# Instructions
- Rewrite the text entirely in first person (jeg, mig, min, etc.).
- Keep the original meaning and wording as close as possible.
- Do not add anything to the text, but change the third person to the first person.
- Resolve all pronouns accordingly.
- Output text in Danish.
- Do not add, remove, or comment on anything — output only the rewritten text in Danish. Output only resolved text, nothing else! No comments, no additional notes, etc.

# Input text:
{dilemma}"""

        return [
            {
                "role": "user",
                "content": prompt
            }
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
        f"first_person_extended_dilemma_{modelname}": pred
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="First Person Rewritting with openai"
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
        default="solutions_dilemmas_all.json",
        help="path to json with all aggregated data in it",
    )

    parser.add_argument(
        "--dilemma_colname",
        type=str,
        default="solutions_dilemmas_all.json",
        help="path to json with all aggregated data in it",
    )


    parser.add_argument(
        "--save_dir",
        type=int,
        default="first_person_rewriting_output",
        help="savedir, where to save predictions",
    )

    return parser.parse_args()

def run(args):
    df = pd.read_json(args.solutions_dilemmas_all_path)

    meta_data = [{"idx": i} for i in df.index.values.tolist()]

    print("Preparing prompts...")
    input_prompts = [
        PersonaResolutionPrompter.prepare_section_messages(dilemma) for dilemma in
        df[args.dilemma_colname].values.tolist()
    ]
    print("Prompts prepared.")

    client = OpenAI(base_url=args.base_url)
    max_workers = 8

    os.makedirs(args.sa, exist_ok=True)

    samples_save_dir = os.path.join(args.save_dir, "generated_samples")
    os.makedirs(samples_save_dir, exist_ok=True)

    run_generations(
        client=client,
        conversations=input_prompts,
        savedir=samples_save_dir,
        meta_data=meta_data,
        modelname=args.modelname,
        max_workers=max_workers
    )

    print("====== Generations done ======")

    gpt_df = agg_data_into_json(args.save_dir)
    gpt_df = pd.concat([gpt_df.drop('meta_data', axis=1), pd.json_normalize(gpt_df['meta_data'])], axis=1)

    df = pd.merge(gpt_df, df, left_on='idx', right_index=True, how='left')

    df.to_json(
        os.path.join(args.save_dir, "solutions_dilemmas_fper.json"),
    )
    print("====== All done ======")


if __name__ == '__main__':
    run(parse_args())
