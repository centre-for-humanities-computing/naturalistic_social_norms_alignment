from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json, os
from tqdm import tqdm
import dirtyjson
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams


def parse_via_dirtyjson(out):
    try:
        out = dirtyjson.loads(out)
        return out
    except:
        return {}


def load_json(out):
    if not isinstance(out, dict):
        if 'assistantfinal' in out:
            out = out.split('assistantfinal')[1]

        try:
            out = json.loads(out)
            return out
        except:
            parsed_out = parse_via_dirtyjson(out)
            return parsed_out

    return out


def parse_matcher(raw_pred):
    if not isinstance(raw_pred, str):
        return raw_pred

    parsed_out = load_json(raw_pred)
    if len(parsed_out):
        return parsed_out

    try:
        if '''"match_results"''' in raw_pred:

            last_advised_occ = [i for i in range(len(raw_pred)) if
                                raw_pred[i:i + len('''"match_results"''')] == '''"match_results"''']
            advised_idx = last_advised_occ[-1]

            all_occur_bracket = [i for i in range(len(raw_pred)) if raw_pred[i] == "{" and i <= advised_idx]

            if all_occur_bracket:
                to_parse_pred = raw_pred[all_occur_bracket[-1]:].strip()
                if to_parse_pred.endswith("```"):
                    to_parse_pred = to_parse_pred[:-len("```")].strip()
                if to_parse_pred.endswith("```"):
                    to_parse_pred = to_parse_pred[:-len("```")].strip()
                return json.loads(to_parse_pred)
            return {}
        return {}
    except:
        return {}


class RuleAnaylsis(BaseModel):
    core_action: str
    participants: str
    details_and_modifiers: str
    conditions: str
    action_order: str
    motivations: str
    pronoun_resolution: str
    strategic_action_equivalence: str


class MatchItem(BaseModel):
    target_solution_id: str
    rule_analysis: RuleAnaylsis
    is_match: bool


class MatchOutput(BaseModel):
    match_results: List[MatchItem]


class MatcherPrompter:
    taskname = 'solution_item_to_list'

    @classmethod
    def prepare_prompt(cls, dilemma, solution_a, solutions_list, name):
        target_solutions = [
            {"content": s['content'], "solution_id": s['solution_id']} for s in solutions_list
        ]

        if isinstance(name, str):
            if "," in name:
                name_prompt = "Vi er " + " og ".join(name.split(",")) + ". "
            else:
                name_prompt = "Jeg er " + name + ". "
        else:
            name_prompt = ""

        prompt_template = """You are an evaluation assistant.
You are provided with:
    - Dilemma
    - A proposed solution to this dilemma
    - A list of target solutions to this dilemma

Your task is to determine, for EACH target solution independently, whether it MATCHES the proposed solution.
You must apply the matching rules strictly and mechanically.

Your goal is NOT to judge quality.
Your goal is NOT to judge usefulness.

You MUST NOT:
- Infer missing details
- Add implicit assumptions
- Use the dilemma to fill missing information
- Generalize
- Assume intent beyond what is explicitly written

## DEFINITION OF A SOLUTION

A solution is a fully specified proposed action(s), including:
- The core action(s)
- Participants involved. Participants are people, organizations, entities and so on. All participants:
    - Who should perform the action
    - Toward whom or what the action is directed
    - Any involved organizations or entities
- All explicitly stated details or modifiers that affect WHAT is done or HOW it is done
- All explicitly stated conditional parts, if any
- Order of actions

A solution is NOT just a verb or a high-level idea. It must capture everything explicitly stated — no more, no less.

IMPORTANT: Motivations, reasons, or justifications for WHY an action should be taken are NOT part of the solution specification and should be IGNORED when comparing solutions.

## DEFINITION OF MATCHING

Two solutions MATCH if and only if they describe:
- The same core action(s)
- Performed by the same participant(s)
- Toward the same target(s)
- With the same explicit details and modifiers that affect WHAT or HOW
- Under the same explicit conditions
- Have the same order of actions

Any missing, added, or changed details aabove (except motivations/reasons) means the solutions DO NOT match.

If one solution expresses a decision (e.g., allowing, refusing, agreeing), and the other expresses a communicative act that directly conveys or implements that same decision, they are considered strategically equivalent.
If two solutions communicate the same outcome, minor modal differences (e.g., "må ikke" vs "kan ikke") do not break strategic equivalence when the practical effect is identical.
Synonyms DO NOT introduce a new action. For example, snakke, diskutere, have en samtale should be considered the same action.


## PRONOUN RESOLUTION RULE

If a solution uses pronouns such as:
- du / I / dig / din / dine
- han / hun
- de / dem / deres
etc.

They must be resolved strictly to explicitly stated participants in the dilemma.

If:
"Du/I" clearly refers to the dilemma narrator and the target explicitly names that narrator, then this counts as the SAME participant.

Pronoun vs explicit name is NOT a mismatch
IF they clearly refer to the same entity.

However:
If a target introduces an additional participant
who is not included in the proposed solution,
this IS a mismatch.


## STRATEGIC ACTION RULE

Count only independent strategic decisions.

Do NOT count as separate actions:

- Inevitable follow-up behavior
- Explicit clarification of implicit co-participants
- Logical consequences of the main action
- Restatements of the same decision in different wording

If a target adds only entailed consequences of the same core decision,
this is still MATCH.


## MATCHING RULES WITH EXAMPLES

In this section you are presented with the example of each rule.

### Core action equivalence

MATCH: "Skriv et brev" / "Forfat et brev"

NOT MATCH: "Skriv et brev" / "Send en sms"

### All explicit details must match (except motivations)

MATCH: "Skriv et brev og tilføj et lille håndtegnet hjerte" / "Skriv et brev med et lille håndtegnet hjerte"

NOT MATCH: "Skriv et brev" / "Skriv et brev og tilføj et lille håndtegnet hjerte"

### Participants must match exactly

MATCH: "Ring til din mor" / "Ring til din mor"

NOT MATCH: "Ring til din mor" / "Ring til din far"

### No missing participants

NOT MATCH: "Ring" / "Ring til din mor"

### Conditions must match

MATCH: "Køb et æble, hvis du ikke er allergisk" / "Hvis du ikke er allergisk, køb et æble"

NOT MATCH: "Køb et æble" / "Køb et æble, hvis du ikke er allergisk"

### Preserved order of actions
MATCH: "Køb et æble og spis det" / "Du kan købe æblet, så du kan spise det bagefter"

NOT MATCH: "Køb et æble eller køb et stykke bacon" / "Køb et æble og køb derefter et stykke bacon"

### Motivations and reasons are ignored

MATCH: "Klæd dig på for din egen skyld" / "Klæd dig på for andres skyld"

MATCH: "Undskyld fordi det er det rigtige at gøre" / "Undskyld for at bevare forholdet"

NOT MATCH (different action despite same motivation): "Ring til hende for at undskylde" / "Skriv til hende for at undskylde"


## Additional Examples

Example dilemma: ""
MATCH (basically the same action): ""

IMPORTANT CONSTRAINTS

- Do NOT assume missing details.
- Do NOT use the dilemma to infer participants, intentions, or conditions.
- IGNORE motivations, reasons, or justifications (e.g., "for your own sake", "because X", "in order to Y").
- Differences in wording are acceptable ONLY if all explicit details match exactly.
- If in doubt, choose NOT MATCH.


OUTPUT FORMAT (STRICT)

Return ONLY the following JSON object, and nothing else:
{{ 
    "match_results": 
        [
            {{
              "target_solution_id": str, // id of the target solution to which the analysis is performed
              "rule_analysis": {{
                "core_action": "MATCH/NOT MATCH - explanation",
                "participants": "MATCH/NOT MATCH - explanation",
                "details_and_modifiers": "MATCH/NOT MATCH - explanation",
                "conditions": "MATCH/NOT MATCH - explanation",
                "action_order": "MATCH/NOT MATCH - explanation",
                "motivations": "IGNORED/N/A - explanation",
                "pronoun_resolution": "MATCH/NOT MATCH - explanation",
                "strategic_action_equivalence": "MATCH/NOT MATCH - explanation"
              }},
              "is_match": bool
            }},
            ...
        ]
}}

Cover ALL the target solutions!

## Examples
### Example 1
Example Dilemma:
Min nabo spiller høj musik hver aften efter kl. 22, og det forstyrrer min søvn. Jeg ved ikke, om jeg skal sige noget eller bare leve med det.

Proposed Solution:
Bank på din nabos dør

Target Solution:
{{
    "content": "Bank på naboens dør",
    "solution_id": "advised-0"
}}

Expected Output: 
{{
  "target_solution_id": "advised-0",
  "rule_analysis": {{
    "core_action": "MATCH - Both solutions describe the same core action: knocking on door",
    "participants": "MATCH - Both refer to the same participant: neighbor (din nabo/naboens)",
    "details_and_modifiers": "MATCH - No additional details or modifiers present in either solution",
    "conditions": "MATCH - No conditions present in either solution",
    "action_order": "MATCH - Single action, no order to compare",
    "motivations": "N/A - No motivations stated in either solution",
    "pronoun_resolution": "MATCH - the actor is the same",
    "strategic_action_equivalence": "MATCH - Both represent the same strategic action with no variation in scope or consequence"
  }},
  "is_match": true
}}

### Example 2
Example Dilemma:
Min teenage søn kommer konstant for sent hjem uden at give besked. Jeg er bekymret for hans sikkerhed, men vil ikke virke kontrollerende. Jeg ved ikke, hvordan jeg skal håndtere det.

Proposed Solution:
Tal med din søn om at sende dig en besked, når han er forsinket

Target Solution:
{{
    "content": "Tal med din søn om at komme hjem til tiden",
    "solution_id": "notadvised-5"
}}

Expected Output:
{{
  "target_solution_id": "notadvised-5",
  "rule_analysis": {{
    "core_action": "NOT MATCH - Proposed Solution: talk about sending notification when delayed. Target Solution: talk about coming home on time. These are fundamentally different actions",
    "participants": "MATCH - Both refer to the same participant: your son (din søn)",
    "details_and_modifiers": "NOT MATCH - Proposed Solution includes 'når han er forsinket' (when delayed), Solution B includes 'til tiden' (on time)",
    "conditions": "MATCH - No conditions in either solution",
    "action_order": "MATCH - Single action in each",
    "motivations": "N/A - No motivations stated",
    "pronoun_resolution": "MATCH - the actor is the same",
    "strategic_action_equivalence": "NOT MATCH - The strategic goal differs: enabling communication when late vs enforcing punctuality"
  }},
  "is_match": false
}}

### Example 3
Example Dilemma:
Min kollega låner ofte mine ting uden at spørge - kuglepenne, hæftemaskine, endda min kaffekrus. Det irriterer mig, men vi skal arbejde tæt sammen på et projekt. Jeg er i tvivl om, hvordan jeg skal tackle situationen uden at ødelægge samarbejdet.

Proposed Solution:
Tag en privat samtale med din kollega og forklar høfligt, at du gerne vil spørges først, og hvis hun er forstående, så tilbyd at dele nogle af tingene fremover

Target Solution:
{{
    "content": "Tag en privat samtale med din kollega og forklar høfligt, at du gerne vil spørges først",
    "solution_id": "advised-13"
}}


Expected Output:
{{
  "target_solution_id": "advised-13",
  "rule_analysis": {{
    "core_action": "NOT MATCH - Both include 'have private conversation and explain politely', but Proposed Solution has an additional conditional action (offer to share)",
    "participants": "MATCH - Both refer to the same participant: your colleague (din kollega)",
    "details_and_modifiers": "MATCH for shared parts - Both include 'privat' (private) and 'høfligt' (politely)",
    "conditions": "NOT MATCH - Proposed Solution includes conditional 'hvis hun er forstående, så tilbyd at dele nogle af tingene' (if understanding, offer to share some things). Target Solution has no conditional part",
    "action_order": "NOT MATCH - Proposed Solution has sequence: conversation → conditional offer. Target Solution only has conversation",
    "motivations": "N/A - No motivations stated",
    "pronoun_resolution": "MATCH - the actor is the same",
    "strategic_action_equivalence": "NOT MATCH - Proposed Solution introduces an additional strategic concession step beyond the core confrontation"
  }},
  "is_match": false
}}

### Example 4
Example Dilemma:
Jeg er midt i 50'erne og sidder i toget på en varm dag iført en undertrøje og en t-shirt uden BH, fordi det er mest behageligt for mig. En kvinde overfor stirrede længe, rejste sig pludselig og sagde til mig, at det generer hende, at mine brystvorter stritter, og at jeg burde tage en trøje på.

Proposed Solution:
{{
    "content": "Du skal klæde dig på for din egen skyld",
    "solution_id": "advised-7"
}}


Target Solution:
Du skal klæde dig på for andres skyld

Expected Output:
{{
  "target_solution_id": "advised-7",
  "rule_analysis": {{
    "core_action": "MATCH - Both solutions describe the same core action: getting dressed (klæde dig på)",
    "participants": "MATCH - Both performed by the same participant: you (du)",
    "details_and_modifiers": "MATCH - No modifiers affecting WHAT or HOW to get dressed",
    "conditions": "MATCH - No conditions in either solution",
    "action_order": "MATCH - Single action",
    "motivations": "IGNORED - 'for din egen skyld' vs 'for andres skyld' are motivations/reasons and do not affect the action itself",
    "pronoun_resolution": "MATCH - the actor is the same",
    "strategic_action_equivalence": "MATCH - The strategic action (getting dressed) is identical; only motivation differs"
  }},
  "is_match": true
}}

### Example 5

Example Dilemma:
Min chef kritiserer ofte mit arbejde foran hele teamet. Det gør mig utilpas og usikker, men jeg vil heller ikke skabe konflikt. Jeg overvejer, om jeg skal sige noget direkte til hende eller bare acceptere situationen.

Proposed Solution:
Tag en privat samtale med din chef om kritikken.

Target Solution:
{{
"content": "Tag en privat samtale med din chef om kritikken og vær forberedt på at svare på hendes spørgsmål",
"solution_id": "advised-21"
}}

Expected Output:
{{
  "target_solution_id": "advised-21",
  "rule_analysis": {{
    "core_action": "MATCH - Both solutions prescribe the same strategic decision: having a private conversation with the boss about the criticism. The target adds preparation for questions, which is an inevitable consequence of initiating such a conversation.",
    "participants": "MATCH - Both involve the same actor (you) and the same target (your boss).",
    "details_and_modifiers": "MATCH - Both specify that the conversation should be private and about the criticism. The added phrase about answering questions does not change the core action.",
    "conditions": "MATCH - No explicit conditions in either solution.",
    "action_order": "MATCH - The main strategic action is identical. The additional clause in the target describes an entailed follow-up behavior, not a separate independent strategic step.",
    "motivations": "N/A - No motivations stated in either solution.",
    "pronoun_resolution": "MATCH - 'din chef' refers to the same explicitly defined person in both solutions.",
    "strategic_action_equivalence": "MATCH - The added preparation is entailed by the same strategic action and does not change its scope"
  }},
  "is_match": true
}}


### Example 6

Example Dilemma:
Mit navn er Anna. Min mand snorker, hvilket gør min søvn virkelig dårlig. Jeg er konstant træt og sover ikke godt. Jeg ved ikke, hvordan jeg skal håndtere dette problem ordentligt, da det virkelig vil skade min mand. Hvad kan jeg gøre?

Proposed Solution:
Før din mand til at gå til lægen angående dette problem, hvis han accepterer lægebesøget.

Target Solution:
{{
"content": "Hvis hendes mand er okay med det, bør Anna tage ham til et hospital for at få undersøgt snorkenproblemet for at forbedre hendes søvn.",
"solution_id": "notadvised-278"
}}

Expected Output:
{{
  "target_solution_id": "notadvised-278",
  "rule_analysis": {{
    "core_action": "MATCH - Both solutions describe the same actions: visiting a medical professional.",
    "participants": "MATCH - Both involve the same actors: Anna and her husband.",
    "details_and_modifiers": "MATCH - Both solutions include the same details.",
    "conditions": "MATCH - In both solutions there is a condition that the husband should be ob board with the decision.",
    "action_order": "MATCH - The order of action is the same in both solutions: go to the medical profesional.",
    "motivations": "IGNORED - In the target solution, there is a motivation to improve Anna's sleep, which is ingored.",
    "pronoun_resolution": "MATCH - The proposed solution is addressed to Anna, same as second solution.",
    "strategic_action_equivalence": "MATCH - The action is the same in both solutions."
  }},
  "is_match": true
}}


### Example 7
Example Dilemma:
Min medarbejder har bedt om at arbejde hjemmefra om fredagen. Jeg er åben for det, men kun hvis hun stadig deltager i alle online møder og leverer sine opgaver til tiden.

Proposed Solution:
Fortæl din medarbejder, at hun gerne må arbejde hjemme om fredagen, så længe hun deltager i møderne og afleverer sine opgaver til tiden.
Target Solution:
{{
"content": "Tillad medarbejderen at arbejde hjemme om fredagen, hvis hun deltager i alle online møder og afleverer sine opgaver til tiden.",
"solution_id": "advised-234"
}}

Expected Output:
{{
  "target_solution_id": "advised-7",
  "rule_analysis": {{
    "core_action": "MATCH - Both solutions express the same core strategic decision: granting permission to work from home under specific performance conditions. The proposed solution frames it as a communicative act, but it conveys the same permission.",
    "participants": "MATCH - In both solutions, the employer is the decision-maker and the employee is the beneficiary of the permission. No strategic participants differ.",
    "details_and_modifiers": "MATCH - Both include identical conditions: participation in meetings and timely task completion.",
    "conditions": "MATCH - The same performance requirements apply in both.",
    "action_order": "MATCH - Single strategic action with attached conditions; no sequencing difference.",
    "motivations": "N/A - No explicit motivations are stated.",
    "pronoun_resolution": "MATCH - References clearly resolve to the same employer and employee roles.",
    "strategic_action_equivalence": "MATCH - Communicating the permission is an entailed implementation of granting it. Both solutions produce the same permitted outcome under the same constraints."
  }},
  "is_match": true
}}

## INPUT

Dilemma:
{name_prompt}{dilemma}

Proposed Solution:
{solution_a}

Target Solutions:
{target_solutions}"""

        return prompt_template.format(
            dilemma=dilemma,
            solution_a=solution_a,
            target_solutions=target_solutions,
            name_prompt=name_prompt
        )


class VLLMBatchProcessor:
    """
    Batch processor for vLLM with structured output support and progress tracking.
    """

    def __init__(
            self,
            model_name: str,
            pipeline_parallel_size: int = 2,
            tensor_parallel_size: int = 1,
            gpu_memory_utilization: float = 0.9,
            max_model_len: Optional[int] = None,
            dtype: str = "auto",
            trust_remote_code: bool = True,
            max_num_seqs=256
    ):
        """
        Initialize vLLM engine.

        Args:
            model_name: HuggingFace model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization fraction
            max_model_len: Maximum sequence length
            dtype: Data type (auto, float16, bfloat16)
            trust_remote_code: Whether to trust remote code
        """
        self.model_name = model_name

        self.llm = LLM(
            model=model_name,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            max_num_seqs=max_num_seqs,
        )

    def generate_batch(
            self,
            prompts: List[str],
            sampling_params: SamplingParams,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of prompt strings
            sampling_params: Sampling parameters

        Returns:
            List of generated text responses
        """
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def check_if_needs_run(self, force_rewrite, checkpoint_path):
        if force_rewrite or not os.path.exists(checkpoint_path):
            return True

        data = json.load(open(checkpoint_path, 'r'))
        if 'error' in data:
            return True

        if not isinstance(data['parsed'], dict):
            return True
        if 'match_results' not in data['parsed']:
            return True

        match_results = data['parsed']['match_results']
        match_results = [x for x in match_results if x['is_match']]
        if len(match_results) >= 3:
            return True

        return False

    def process_with_checkpoint(
            self,
            prompts: List[str],
            metadatas: List[Dict[str, Any]],
            save_dir: str,
            batch_size: int = 32,
            sampling_params: Optional[SamplingParams] = None,
            response_format: Optional[BaseModel] = None,
            parse_function: Optional[callable] = None,
            force_rewrite: bool = False,
            description: str = "Processing"
    ):
        os.makedirs(save_dir, exist_ok=True)

        # Determine which samples need processing
        indices_to_process = []
        for i, metadict in enumerate(metadatas):

            checkpoint_path = f"""Dilemma{metadict['dilemma_id']} -- Pred{metadict['pred-solution-id']}.json"""
            checkpoint_path = os.path.join(save_dir, checkpoint_path)
            if self.check_if_needs_run(force_rewrite, checkpoint_path):
                indices_to_process.append(i)

        print(f"{description}: Processing {len(indices_to_process)} / {len(prompts)} samples")

        # Process in batches

        for batch_start in tqdm(
                range(0, len(indices_to_process), batch_size),
                desc=description
        ):
            batch_end = min(batch_start + batch_size, len(indices_to_process))

            # These are the actual original indices
            batch_indices = indices_to_process[batch_start:batch_end]

            batch_prompts = [prompts[i] for i in batch_indices]
            batch_metadatas = [metadatas[i] for i in batch_indices]

            try:
                batch_outputs = self.generate_batch(batch_prompts, sampling_params)
            except Exception as e:
                print(f"Error generating batch {batch_start}-{batch_end}: {e}")
                for prompt_i, meta in zip(batch_indices, batch_metadatas):
                    self._save_result(save_dir, prompts[prompt_i], None, meta, error=str(e))
                continue

            for prompt_idx, output_text, metadata in zip(
                    batch_indices, batch_outputs, batch_metadatas
            ):
                parsed_output = self._parse_output(output_text, response_format, parse_function)
                self._save_result(save_dir, prompts[prompt_idx], parsed_output, metadata)

    def _load_all_results(
            self, save_dir: str, expected_fienames
    ):
        """Load all results from disk in order."""
        results = []
        for filename in expected_fienames:
            checkpoint_path = os.path.join(save_dir, filename)
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, "r") as f:
                    results.append(json.load(f))
            else:
                results.append({
                    "parsed": None,
                    "prompt": None,
                    "metadata": None,
                    "error": "Missing checkpoint"
                })
        return results

    def _parse_output(
            self,
            output_text: str,
            response_format: Optional[BaseModel] = None,
            parse_function: Optional[callable] = None,
    ) -> Any:
        """
        Parse output text to structured format.

        Args:
            output_text: Generated text
            response_format: Pydantic model for structured output
            parse_function: Fallback parsing function

        Returns:
            Parsed output (dict or original text)
        """
        if response_format is None:
            return output_text

        try:
            # Try to parse as JSON matching the schema
            parsed = json.loads(output_text)
            # Validate with Pydantic
            validated = response_format(**parsed)
            return validated.dict()
        except Exception as e:
            # Fallback to custom parse function if provided
            if parse_function is not None:
                try:
                    return parse_function(output_text)
                except Exception as parse_e:
                    print(f"Parse function failed: {parse_e}")

            print(f"Failed to parse output: {e}")
            return None

    def _save_result(
            self,
            save_dir: str,
            prompt: str,
            parsed_output: Any,
            metadata: Dict[str, Any],
            error: Optional[str] = None,
    ):
        """Save a single result to disk."""
        result = {
            "parsed": parsed_output,
            "prompt": prompt,
            "metadata": metadata,
        }
        if error:
            result["error"] = error

        savefilename = f"""Dilemma{metadata['dilemma_id']} -- Pred{metadata['pred-solution-id']}.json"""
        checkpoint_path = os.path.join(save_dir, savefilename)
        with open(checkpoint_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            # print("--> Saved to: ", checkpoint_path)


def run_parallel_issue_fixing(
        processor: VLLMBatchProcessor,
        prompts: List[str],
        savedir: str,
        taskname: str,
        batch_size: int,
        response_format: BaseModel,
        parse_function: callable,
        n_issue_resolve_attempts: int,
        metadatas: Optional[List[Dict]] = None
):
    """
    Run batch processing with automatic retry for parsing failures.

    Args:
        processor: VLLMBatchProcessor instance
        prompts: List of prompts
        savedir: Directory to save results
        taskname: Name of the task (subdirectory)
        batch_size: Batch size for processing
        response_format: Pydantic model for structured output
        parse_function: Function to parse malformed outputs
        n_issue_resolve_attempts: Number of retry attempts for failed parses
        metadatas: Optional metadata for each prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    """
    print(f"[{taskname}] >>>> Start!")
    task_savedir = os.path.join(savedir, taskname)
    os.makedirs(task_savedir, exist_ok=True)
    from vllm.sampling_params import StructuredOutputsParams
    json_schema = response_format.model_json_schema()

    guided_decoding_params = StructuredOutputsParams(json=json_schema)

    # Create sampling params
    sampling_params = SamplingParams(
        max_tokens=None,
        structured_outputs=guided_decoding_params
    )

    # Initial processing
    processor.process_with_checkpoint(
        prompts=prompts,
        metadatas=metadatas,
        save_dir=task_savedir,
        batch_size=batch_size,
        sampling_params=sampling_params,
        response_format=response_format,
        parse_function=parse_function,
        force_rewrite=False,
        description=f"Running {taskname}"
    )

    expected_fienames = [
        f"""Dilemma{metadict['dilemma_id']} -- Pred{metadict['pred-solution-id']}.json""" for metadict in metadatas
    ]
    outputs = processor._load_all_results(
        task_savedir, expected_fienames
    )

    # Identify and retry failed parses
    for attempt in range(n_issue_resolve_attempts):
        issue_indices = [
            i for i, output in enumerate(outputs)
            if output["parsed"] is None or output["parsed"] == {}
        ]

        if not issue_indices:
            print(f"[{taskname}] All issues resolved!")
            break

        print(
            f"[{taskname}] Attempt {attempt + 1}: "
            f"{len(issue_indices)} issues remaining"
        )

        # Retry with higher temperature to get different outputs
        retry_sampling_params = SamplingParams(max_tokens=None)
        # Create subset for retry
        retry_prompts = [prompts[i] for i in issue_indices]
        retry_metadatas = [metadatas[i] for i in issue_indices]

        # Process with force rewrite only for failed indices
        processor.process_with_checkpoint(
            prompts=retry_prompts,
            metadatas=retry_metadatas,
            save_dir=task_savedir,
            batch_size=batch_size,
            sampling_params=retry_sampling_params,
            response_format=response_format,
            parse_function=parse_function,
            force_rewrite=True,
            description=f"[{taskname}] Resolving issues (attempt {attempt + 1})"
        )
        outputs = processor._load_all_results(
            task_savedir, expected_fienames
        )

    print(f"[{taskname}] >>>> Done!")


def prepare_prompts(df, target_solutions_colname, pred_solutions_colname, dilemma_colname, author_name_colname):
    metadata = []
    prompts = []

    for i in range(len(df)):
        if df.iloc[i][pred_solutions_colname]:
            for pred_solution_dict in df.iloc[i][pred_solutions_colname]:
                metadata.append(
                    {
                        "pred-solution-id": pred_solution_dict['solution_id'],
                        'dilemma_id': df.iloc[i]['dilemma_id']
                    }
                )

                prompts.append(
                    MatcherPrompter.prepare_prompt(
                        dilemma=df.iloc[i][dilemma_colname],
                        solution_a=pred_solution_dict['content'],
                        solutions_list=df.iloc[i][target_solutions_colname],
                        name=df.iloc[i][author_name_colname]
                    )
                )

    return prompts, metadata


import pandas as pd
import os, json


def find_matched(match_dicts, expected_target_preds):
    matched_ids = []

    for match_dict in match_dicts:
        if not isinstance(match_dict['is_match'], bool):
            print("---> Not bool: ", match_dict)
        if match_dict['is_match']:
            if match_dict['target_solution_id'] in expected_target_preds:
                matched_ids.append(match_dict['target_solution_id'])
    return matched_ids


def load_single_model_preds(dirpath, modelname, meta_df, reference_solutions_colname):
    output = []

    for filename in os.listdir(dirpath):
        if filename.endswith(".json"):
            file_content = json.load(
                open(os.path.join(dirpath, filename), 'r')
            )

            dilemma_id, pred_id = filename.split(" -- Pred")
            dilemma_id = dilemma_id[len("Dilemma"):]

            pred_id = pred_id.replace(".json", "")

            expected_target_preds = meta_df[meta_df['dilemma_id'] == dilemma_id].iloc[0]
            expected_target_preds = [x['solution_id'] for x in
                                     expected_target_preds[reference_solutions_colname]]

            matched_targets = find_matched(file_content['parsed']['match_results'], expected_target_preds)

            out_dict = {
                "model": modelname,
                "pred_id": pred_id,
                'dilemma_id': dilemma_id,
                'matched_targets': matched_targets
            }
            output.append(out_dict)
    return output


def load_match_preds_to_df(loaddir, meta_df, reference_solutions_colname):
    outputs = []

    for dirname in os.listdir(loaddir):
        print(dirname)
        full_dirname = os.path.join(loaddir, dirname)
        if os.path.isdir(full_dirname):
            matched_outputs_path = os.path.join(full_dirname, 'solution_item_to_list')

            outputs += load_single_model_preds(matched_outputs_path, dirname, meta_df, reference_solutions_colname)

    return pd.DataFrame(outputs)



if __name__ == '__main__':
    import pandas as pd

    df = pd.read_json("processed_extracted_solutions_df.json")

    BATCH_SIZE = 2_500
    N_FORMAT_ATTEMPTS = 150


    reference_solutions_colname = "target_matched_postprocessed_solutions_first_person"
    dilemma_colname = "first_person_dilemma"
    author_name_colname = "author_name"

    matching_processor = VLLMBatchProcessor(
        model_name="google/gemma-3-27b-it",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.94,
        pipeline_parallel_size=1,
        max_model_len=15_000,
        max_num_seqs=1024
    )

    for pred_solutions_colname in [
        'odin-large -- postprocessed_extracted_solutions_first_person'
    ]:
        savedir = "matched_preds/"
        os.makedirs(savedir, exist_ok=True)

        savedir = os.path.join(savedir, pred_solutions_colname.split(" -- ")[0].strip().replace("/", "-"))
        os.makedirs(savedir, exist_ok=True)

        print("Saving to: ", savedir)

        match_prompts, metadatas = prepare_prompts(
            df=df,
            target_solutions_colname=reference_solutions_colname,
            pred_solutions_colname=pred_solutions_colname,
            dilemma_colname=dilemma_colname,
            author_name_colname=author_name_colname
        )
        print(f"Num prompts to run: {len(match_prompts)}")

        run_parallel_issue_fixing(
            processor=matching_processor,
            prompts=match_prompts,
            savedir=savedir,
            taskname=MatcherPrompter.taskname,
            batch_size=BATCH_SIZE,
            response_format=MatchOutput,
            parse_function=parse_matcher,
            n_issue_resolve_attempts=N_FORMAT_ATTEMPTS,
            metadatas=metadatas
        )


    final_df = load_match_preds_to_df(
        loaddir=os.path.join(loaddir, MatcherPrompter.taskname),
        meta_df=df,
        reference_solutions_colname=reference_solutions_colname
    )
    final_df.to_json("matched_df.json")
