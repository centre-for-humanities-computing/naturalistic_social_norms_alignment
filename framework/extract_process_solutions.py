import os
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import pandas as pd
from pydantic import BaseModel
from typing import List
import nltk, json

nltk.download('punkt')
nltk.download('punkt_tab')
import os
from nltk import sent_tokenize
from tqdm import tqdm
from itertools import combinations
import dirtyjson
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import networkx as nx


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


def parse_extraction(raw_pred):
    if not isinstance(raw_pred, str):
        return raw_pred

    parsed_out = load_json(raw_pred)
    if len(parsed_out):
        return parsed_out

    try:
        if '''"advised"''' in raw_pred:

            last_advised_occ = [i for i in range(len(raw_pred)) if
                                raw_pred[i:i + len('''"advised"''')] == '''"advised"''']
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


def parse_matcher(raw_pred):
    if not isinstance(raw_pred, str):
        return raw_pred

    parsed_out = load_json(raw_pred)
    if len(parsed_out):
        return parsed_out

    try:
        if '''"rule_analysis"''' in raw_pred:

            last_advised_occ = [i for i in range(len(raw_pred)) if
                                raw_pred[i:i + len('''"rule_analysis"''')] == '''"rule_analysis"''']
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


class FinalSolutions(BaseModel):
    advised: list[str]
    not_advised: list[str]


def prepare_answer_input(answer_text):
    segments_str_input = ""
    for i, sent in enumerate(sent_tokenize(answer_text)):
        segments_str_input += f"""\nID: {i}; Speaker 0: {sent.strip()}\n"""

    segments_str_input = segments_str_input.strip()
    return segments_str_input


class ExtractionPrompter:
    taskname = 'solution_extraction'

    @classmethod
    def _prepare_prompt(cls, transcript, dilemma):
        prompt = """You are a helpful solution extraction assistant. You are presented with the podcast transcript called "Sara og Monopolet" (or "Mads og Monopolet"), where the guests are presented with the dilemma from the listener. The guests are then brainstorming multiple solutions that are either advised or not advised to do for the listener.
You are also provided with a dilemma itself. The transcript can contain irrelevant discussions (usually at the beginning of the conversation), but you need to focus ONLY on the discussion that is relevant to the provided dilemma.

Your objective is to:
- Extract solutions that the guests of the podcast came up with
- Extract all the solutions that were advised to do; 
- Extract all the solutions that were not advised to do

# Output format
Generate output in the following format:
```json
{
    "advised": [
        str, // solution text
        str, // solution text
        ...
    ],
    "not_advised": [
        str, // solution text
        str, // solution text
        ...
    ]
}
```

# Solution Generation Guidelines
## Avoid too general solutions
This rule is applicable for both advised and not_advised solutions.
If the solution is too general and do NOT propose any specific actions, do not add it to the final solution list. For example, in the solution "You have to do improve your life", no specific actions are provided, therefore it is too general and has to be ignored.

## Remove any emotional or manipulative styles from the solution
This rule is applicable for both advised and not_advised solutions.
Do not include appeal to emotion in the solution, replace it with the more neutral text. For example, the text "Du bør lade andre gå over dig." contains a manipulative elements, so you need to rewrite it to something like "Du bør overveje at hjælpe andre og følge deres anvisninger.".
If in the in the solution there is something that should not be taken literally, you need to rewrite so it will reflect the meaning of it. Do not use provers etc, the meaning should be clear and neutral. For example,  "Du burde gå med til at begrave stridsøksen efter års skænderier med din bror.", from the context you need to figure out if it is a literal action or just a common saying. If it is a common saying, rewrite it to "Du bør gå med til at slutte fred med din bror."
Do not include any negative or positive emotions in the solution. Make sure that the solution sounds as neutral as possible, so the reader would not have a bias towards it.

## What is advised and what is not_advised
- *advised*: Solutions that were mentioned in the input text as advised in favor of etc.
- *not_advised*: Solutions that were mentioned in the input text as advised against etc. If the solution contains a negative negation words (e.g. "ikke") *towards action* or there is an indication to NOT do some action, it is not advised solutions.
*CRUCIAL*: Do not mess them up! If in the advised solution you extracted negative negation towards an action (e.g. words like Avoid, Do not, etc), you have made a mistake. Fix it and assign this solution as not_advised.

## not_advised solution generation
If in the text you have encountered a solution that is advised against, you need to remove all negative negations from it.  Rewrite it by stripping the indirect_way and direct_way from the negation words. For example, if in the text it is stated "Spis ikke æblet", you need to extract the solution "Spis æblet". For example, writing "Undgå at spise æbler" should be written as "Spis æblet".

## Very Important: Solution Content and Granularity Guidelines
- Extract all relevant information about the solution:
    - Extract all the details for the solution. For example, if the text stated "Jeg synes, du skal skrive et brev. Breve er en god måde at kommunikere på, og tilføj et lille håndtegnet hjerte. Jeg er sikker på, at han vil kunne lide det.", extracting just "du skal skrive et brev" is not enough. You need to extract something like: "Du skal skrive et brev og inkludere et lille håndtegnet hjerte."
    - Participants of the solution (who should perform it, towards whom, which organizations/entities are involved etc.). For example, generating "Ring" is incorrect as it is missing a participant: (call whom ?). The correct version would be "Ring til din mor"
    - Avoid using pronouns except for "Du" or "De". For example, writing "Ring til ham" is incorrect, the correct version would be "Ring til din far". In this case, it is acceptable to use pronoun "din", but not "ham".
    - Conditional parts if they are mentioned int he input text for this particular solution (e.g. "Køb et æble, hvis du ikke har allergi over for det"). If the text does not mention any conditional part explicitly, do not add it to the solution.
    - Order of actions. For example, in the text "ring til din far og ring derefter til din mor", you need to extract it as a single solution, NOT as two solutions: "ring til din far", "ring til din mor". Moreover, the order has to be preserved the same as it stated in the transcript or the dilemma!
    - If the solution contains a negative negation words (e.g. "ikke") *towards action* or there is an indication to NOT do some action, it should be extracted as *not_advised* and rewritten accordingly: strip them from the negation words. For example, if in the text it is stated "Spis ikke æblet", you need to extract the solution "Spis æblet" undernot_advised*. For example, writing "Undgå at spise æbler" should be written as "Spis æblet" undnot_advised.
   - Do not include any negative negation marks *towards action* (e.g. words like "ikke") in the solutions! If the solution contains the negative negation mark, you did it wrong: if it is under *not_advised*, you need to remove it there. 
- Do not imagine solutions, only extract the ones that were explicitly stated in the text or the dilemma.
- Do not include any mentions about who suggested the solution or any other information about the speaker. Extract ONLY the solution itself.
- Extract ALL of the solutions, even if you do not like them or disagree with them.
- Avoid too general solutions like "Do something".
- Solutions have to be written in the natural Danish.
- Do not incorporate any structure in the solution text: make it natural and compliant with all the requirement above.

*CRUCIAL*: All the solutions should be processed! DO NOT OMIT ANY SOLUTIONS!

## Examples
Input Dilemma
Mit navn er Ann, og jeg har en ven, der altid kommer for sent. Han har virkelig hjulpet mig meget tidligere, han var der for mig i de svære tider i mit liv. Vi har kendt hinanden i 10 år nu. Han har dog en forfærdelig vane: han glemmer normalt at sætte de beskidte tallerkener i vasken, når han besøger mig. Han smider dem bare ud over det hele, hvilket er virkelig irriterende. Hvordan kan jeg håndtere det? Skal jeg tale med ham? Skal jeg slet ikke gøre noget? Skal jeg spørge hans venner, om han gør det i deres lejlighed, og hvis ja, skal vi konfrontere ham sammen?

Input Transcript
Speaker 1: okay, så eh Ans dilemma, ikke. de har kendt hinanden i ti år, han lyder som en rigtig god ven men alts  tallerkener over det hele

Speaker 2: ja, det er sådan noget der virkelig kan gå en på nerverne. især når det er ens egen lejlighed. og hun siger jo han har hjulpet hende meget før

Speaker 3: mm ja præcis. så jeg synes ikke bare man skal ignorere det for evigt. det ender bare med irritation

Speaker 1: nej enig. men jeg ville heller ikke gøre det til et stort drama. for mig lyder det som en rolig snak er vejen frem

Speaker 2: ja helt klart. og ikke over telefonen. det bliver sådan lidt forkert, eller misforstået

Speaker 3: ja, vent til det sker igen. altså når han faktisk har glemt en tallerken, og I begge er der

Speaker 1: præcis, så kan man sige det i situationen. sådhey, når du gør det her, så går det mig faktis

Speaker 2: mm ja, hold det konkret. ikke du gør alti, men bare at det ikke er okay for dig

Speaker 3: og de er jo tætte venner, så jeg tror faktisk han vil forstå det. det lyder ikke som om han gør det med vilje

Speaker 1: ja, så rolig samtale, ansigt til ansigt, i øjeblikket

Speaker 2: ja, og helt sikkert ikke en telefon-snak

Speaker 3: nej nej, aldrig telefonen

Speaker 1: men altså, hvis han bliver defensiv eller sådan lidt fornærmet, så ville jeg nok først lige trække vejret, og så bagefter forklare hvorfor det faktisk betyder noget for dig

Speaker 2: ja, god pointe. og hvis det fortsætter efter samtalen, så synes jeg først du skal minde ham roligt om det, og hvis det stadig ikke ændrer sig, så kan du tage snakken igen, men mere tydeligt

Speaker 3: mm, ja. og hvis han næste gang kommer på besøg, så kunne du også sige det allerede når han kommer ind, altså først sætte forventningen, og så senere se om han husker det



Expected Output
{
    "advised": [
        "Du kan have en rolig samtale med din ven om hans opførsel.",
        "Når din ven glemmer at sætte tallerkenen i vasken, så forklar ham, at det ikke er acceptabelt.",
        "Hav en samtale med din ven, og hvis din ven bliver defensiv, så tag først en indånding og forklar derefter, hvorfor det er vigtigt for dig.",
        "Snakk med din ven om dette problem. Hvis problemet fortsætter efter samtalen, så mind ham først roligt om det, og hvis det stadig ikke ændrer sig, så tag det op igen mere tydeligt.",
        "Hvis din ven kommer på besøg igen, kan du indstille forventningen, når han ankommer, og så se, om han husker det senere."
        ],
    "not_advised": [
        "Du bø tale med din ven om problemet i telefonen."
        ]
}


"""

        prompt += f"""## Inputs

## Dilemma
{dilemma}

## Transcript
{transcript}"""

        return prompt

    @classmethod
    def prepare_prompts(cls, df, dilemma_colname, answer_colname):
        prompts = []

        for index, row in df.iterrows():
            prompts.append(
                cls._prepare_prompt(
                    transcript=row[answer_colname],
                    dilemma=row[dilemma_colname]
                )
            )

        return prompts


class SolutionPostprocessingPrompter:
    taskname = "postprocessing_solutions"

    @classmethod
    def _prepare_prompt(cls, transcript, dilemma, solutions_dict):
        solutions = json.dumps(solutions_dict, indent=4)

        prompt = """You are presented with the following:
- dilemma: dilemma that you need to focus on. 
- An original open-ended answer to the dilemma.
- A dictionary of extracted proposed solutions. Each solution is marked as advised or not_advised.


Your task is to **postprocess the extracted solutions** according to these rules:
- Carefully read the transcript part which corresponds to the provided dilemma.
- Carefully read a dictionary of extracted proposed solutions.

Postprocess solution via the following rules:

1. **Remove jokes or clearly unrealistic solutions.**

   * Example:
     * “If your colleague criticizes your work, punch them in the face” → joke/unrealistic → remove.
     * “If someone is bullying you, challenge them to a dance-off” → joke → remove.

   * **Do not remove serious advices**. Only remove answers that are clearly intended as a joke or absurd.

2. **Ensure negations are correctly represented.**
For each solution, there are advised and not_advised fields:
- *advised*: Solutions that were mentioned in the input text as advised in favor of, mentioned as the author of the text would do in the position of the dilemma etc.
- *not_advised*: Solutions that were mentioned in the input text as advised against, mentioned as the author of the text would not do in the position of the dilemma etc. If the solution contains a negative negation words (e.g. "ikke") *towards action* or there is an indication to NOT do some action, it is not advised solutions.

A solution must be classified based on its semantic meaning, not only the presence of explicit negation words like “ikke”.

Any solution that expresses refusal, rejection, declining, avoidance, or non-participation is considered a NEGATION of an action, even if no explicit negation word is present.

This includes (but is not limited to) verbs and constructions such as: afvise, sige nej kan ikke hjælpe, holde sig fra, undlade, vælge ikke at, følge reglerne i stedet for at hjælpe, beskytte sig selv mod konsekvenser, and so on

These must be handled exactly like explicit negations.

Read the advised and not_advised solutions.

2.1 **Rewrite not_advised solution if necessary**
If there is a negative negation in the not_advised solution, rewrite it and KEEP it under not_advised. 
If a solution is phrased as a refusal or non-action (e.g. “afvise”, “ikke hjælpe”), rewrite it into the corresponding positive action and keep it under not_advised.

Otherwise, DO NOT change it.

    * Example:
        * Original, advised: “Du skal afvise venindens anmodning om hjælp” → not_advised: “Du skal hjælpe din veninde med eksamen.”
        * Original, advised : “Du kan ikke hjælpe hende uden at risikere problemer” → not_advised: “Du skal hjælpe hende, selvom det kan give problemer.”

If the solution is already a positive action with no negation → do not change it, just keep it under not_advised.

All `not_advised` solutions must be positive actions that are discouraged.

   * Example:
     * Original under not_advised: “Do not go out with Anders” → move to `not_advised` as positive action → “Go out with Anders.”
     * Original under not_advised: “Hide your brothers computer” → no negative negation → do not change, keep it under not_advised

2.2. **Check advised solution on negative negation**
All `advised` solutions must be positive actions. If a solution expresses refusal, avoidance, or non-action, it CANNOT remain in advised, even if phrased as “you should”.
Read advised solution. If it contains negative negation, it should be removed from advised and placed into not_advised with changed negation to positive.
If it DOES NOT contain negative negation, it DO NOT change it and KEEP it under advised. 

   * Example:
     * Original under advised: “Avoid buying this apple” → contains negative negation → move to `not_advised` and change the negation → “Buy this apple.” under not_advised
     * Original under not_advised: “You can keep the laptop and do not tell anybody” → no negative negation → do not change, keep it under advised


When deciding whether a solution should be placed under advised or not advised, apply this priority order:
    - Semantic meaning (what action is encouraged or discouraged)
    - Refusal / non-action semantics
    - Explicit negation words (ikke, undgå, afvise)
    - Grammatical form
If there is any conflict, semantic meaning always wins.

CRUCIAL RULES:
- DO NOT remove ANY details (e.g. names, order of actions, entities, conditional parts etc) from the solutions!
- The output solutions MUST BE in Danish!

# Output format
Generate output in the following format:
```json
{
    "advised": [
        str, // solution str,
        str, // solution str
        ...
        ],
    "not_advised": [
        str, // solution str,
        str, // solution str
        ...
    ]
}
```

## Examples
Input Dilemma
Mit navn er Ann, og jeg har en ven, der altid kommer for sent. Han har virkelig hjulpet mig meget tidligere, han var der for mig i de svære tider i mit liv. Vi har kendt hinanden i 10 år nu. Han har dog en forfærdelig vane: han glemmer normalt at sætte de beskidte tallerkener i vasken, når han besøger mig. Han smider dem bare ud over det hele, hvilket er virkelig irriterende. Hvordan kan jeg håndtere det? Skal jeg tale med ham? Skal jeg slet ikke gøre noget? Skal jeg spørge hans venner, om han gør det i deres lejlighed, og hvis ja, skal vi konfrontere ham sammen?

Input Transcript
Speaker 1: okay, så eh Ans dilemma, ikke. de har kendt hinanden i ti år, han lyder som en rigtig god ven men alts  tallerkener over det hele

Speaker 2: ja, det er sådan noget der virkelig kan gå en på nerverne. især når det er ens egen lejlighed. og hun siger jo han har hjulpet hende meget før

Speaker 3: mm ja præcis. så jeg synes ikke bare man skal ignorere det for evigt. det ender bare med irritation

Speaker 1: nej enig. men jeg ville heller ikke gøre det til et stort drama. for mig lyder det som en rolig snak er vejen frem

Speaker 2: ja helt klart. og ikke over telefonen. det bliver sådan lidt forkert, eller misforstået

Speaker 3: ja, vent til det sker igen. altså når han faktisk har glemt en tallerken, og I begge er der

Speaker 1: præcis, så kan man sige det i situationen. sådhey, når du gør det her, så går det mig faktis

Speaker 2: mm ja, hold det konkret. ikke du gør alti, men bare at det ikke er okay for dig

Speaker 3: og de er jo tætte venner, så jeg tror faktisk han vil forstå det. det lyder ikke som om han gør det med vilje

Speaker 1: ja, så rolig samtale, ansigt til ansigt, i øjeblikket

Speaker 2: ja, og helt sikkert ikke en telefon-snak

Speaker 3: nej nej, aldrig telefonen

Speaker 1: men altså, hvis han bliver defensiv eller sådan lidt fornærmet, så ville jeg nok først lige trække vejret, og så bagefter forklare hvorfor det faktisk betyder noget for dig

Speaker 2: ja, god pointe. og hvis det fortsætter efter samtalen, så synes jeg først du skal minde ham roligt om det, og hvis det stadig ikke ændrer sig, så kan du tage snakken igen, men mere tydeligt

Speaker 3: mm, ja. og hvis han næste gang kommer på besøg, så kunne du også sige det allerede når han kommer ind, altså først sætte forventningen, og så senere se om han husker det

Speaker 2: Hvis han ringer og beder om at besøge dig, synes jeg også, du skal sige nej.

Speaker 3: eller du kan altid gemme tallerkenerne under sofaen og lade som om, der ikke er sket noget, indtil det stinker huset - så flytter du dig bare

Speaker 1: Ja, det vil vise ham


Input Solutions
{
    "advised": [
        "Du bør afvise din ven, når han beder om at besøge din lejlighed igen",
        "Du kan have en rolig samtale med din ven om hans vane",
        "Du bør ikke ignorere denne situation",
        "Du kan have en rolig samtale med din ven om hans vane. Hvis han bliver defensiv eller fornærmet, så tag en dyb indånding, rolig ned og forklar hvorfor det er vigtigt for dig.",
        "Du kan have en rolig samtale med din ven om hans vane. Hvis han efter samtalen fortsætter med at gøre det samme, så tal med ham igen og vær mere specifik",
        Når din ven kommer på besøg, så sæt forventningerne til, hvad reglerne er vedrørende tallerkenerne. Se derefter, om han husker det",
            ],
    "not_advised": [
        "Tag en telefonsamtale med din ven angående dette problem",
        "Du kan gemme opvasken under sofaen og lade som om, at ingenting er sket, indtil huset stinker. Derefter flytter du bare"
        ]
}


Expected Output:
{
    "advised": [
    "Du kan have en rolig samtale med din ven om hans vane",
    "Du kan have en rolig samtale med din ven om hans vane. Hvis han bliver defensiv eller fornærmet, så tag en dyb indånding, ro dig ned og forklar hvorfor det er vigtigt for dig.",
     "Du kan have en rolig samtale med din ven om hans vane. Hvis han efter samtalen fortsætter med at gøre det samme, så tal med ham igen og vær mere specifik",
     "Når din ven kommer på besøg, skal du fastsætte forventningerne til, hvad reglerne er vedrørende tallerkenerne. Se derefter, om han husker det"
        ],
    "not_advised": [
    "Tag en telefonsamtale med din ven angående dette problem",
    "Du bør ignorere "denne situation",
    "Du bør lade din ven komme på besøg, når han spørger om at besøge din lejlighed igen"
        ]
}

Explanation:
In the example, the input solutions should have been modified in the following way:
  "Du bør afvise din ven, når han beder om at besøge din lejlighed igen" contained a negative negation and it was under advised. So, it was rewritten with no negative negation and placed into not_advised.
  "Du bør ikke ignorere denne situation" solution contained a negative negation and was in advised. It was moved to the not_advised, and its negation was changed.
  Solution "Du kan gemme opvasken under sofaen og lade som om, at ingenting er sket, indtil huset stinker. Derefter flytter du bare" is clearly not serious and was a joke
  Solkutions "Tag ikke en telefonsamtale med din ven angående dette problem" had a negative negation and was under not_advised. So, the negation was changed and it was kept under the not_advised.

"""

        prompt += f"""## Inputs

## Dilemma
{dilemma}

## Transcript
{transcript}

## Input Solutions
{solutions}"""

        return prompt

    @classmethod
    def prepare_prompts(cls, df, dilemma_colname, solutions_colname, answer_colname):
        prompts = []
        metadata = []

        for index, row in df.iterrows():
            prompts.append(
                cls._prepare_prompt(
                    transcript=row[answer_colname],
                    dilemma=row[dilemma_colname],
                    solutions_dict=row[solutions_colname]
                )
            )
            metadata.append(row.to_dict())

        return prompts, metadata


class RuleAnaylsis(BaseModel):
    core_action: str
    participants: str
    details_and_modifiers: str
    conditions: str
    action_order: str
    motivations: str


class MatchOutput(BaseModel):
    rule_analysis: RuleAnaylsis
    is_match: bool


class MatcherPrompter:
    taskname = 'solution_pair_matching'

    @classmethod
    def prepare_prompt(cls, dilemma, solution_a, solution_b):

        prompt_template = """You are an evaluation assistant.
You are provided with:
    - Dilemma
    - Two solutions to this dilemma: Solution A and Solution B.

Your task is to determine whether two extracted solutions represent the SAME solution to a given dilemma.

Your decision must be based on exact matching of all relevant solution details, not general similarity.
Follow the rules strictly and do not infer missing information.

## DEFINITION OF A SOLUTION

A solution is a fully specified proposed action, including:
- The core action(s)
- Participants involved. Participants are people, organizations, entities and so on. All participants:
    - Who should perform the action
    - Toward whom or what the action is directed
    - Any involved organizations or entities
- All explicitly stated details or modifiers that affect WHAT is done or HOW it is done
- All explicitly stated conditional parts, if any
- Order of actions, if any

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

Any missing, added, or changed detail (except motivations/reasons) means the solutions DO NOT match.

## MATCHING RULES WITH EXAMPLES

In this section you are presented with the example of each rule in English and Danish.

### Core action equivalence

MATCH:
- EN: "Write a letter" / "Compose a written letter"
- DA: "Skriv et brev" / "Forfat et brev"

NOT MATCH:
- EN: "Write a letter" / "Send a text message"
- DA: "Skriv et brev" / "Send en sms"

### All explicit details must match (except motivations)

MATCH:
- EN: "Write a letter and include a small hand-drawn heart" /
      "Write a letter with a small hand-drawn heart included"
- DA: "Skriv et brev og tilføj et lille håndtegnet hjerte" /
      "Skriv et brev med et lille håndtegnet hjerte"

NOT MATCH:
- EN: "Write a letter" /
      "Write a letter and include a small hand-drawn heart"
- DA: "Skriv et brev" /
      "Skriv et brev og tilføj et lille håndtegnet hjerte"

### Participants must match exactly

MATCH:
- EN: "Call your mother" / "Phone your mom"
- DA: "Ring til din mor" / "Ring til din mor"

NOT MATCH:
- EN: "Call your mother" / "Call your father"
- DA: "Ring til din mor" / "Ring til din far"

### No missing participants

NOT MATCH:
- EN: "Call" / "Call your mother"
- DA: "Ring" / "Ring til din mor"

### Conditions must match

MATCH:
- EN: "Buy an apple if you are not allergic" /
      "If you are not allergic, buy an apple"
- DA: "Køb et æble, hvis du ikke er allergisk" /
      "Hvis du ikke er allergisk, køb et æble"

NOT MATCH:
- EN: "Buy an apple" /
      "Buy an apple if you are not allergic"
- DA: "Køb et æble" /
      "Køb et æble, hvis du ikke er allergisk"

### Preserved order of actions
MATCH:
- EN: "Buy an apple and then eat it" / "You can purchase the apple, so afterwards you can eat it"
- DA: "Køb et æble og spis det" / "Du kan købe æblet, så du kan spise det bagefter"

NOT MATCH:
- EN: "Buy an apple or buy a piece of bacon" / "Buy an apple and after buy a piece of bacon"
- DA: "Køb et æble eller køb et stykke bacon" / "Køb et æble og køb derefter et stykke bacon"

### Motivations and reasons are ignored

MATCH:
- EN: "Get dressed for your own sake" / "Get dressed for others' sake"
- DA: "Klæd dig på for din egen skyld" / "Klæd dig på for andres skyld"

MATCH:
- EN: "Apologize because it's the right thing to do" / "Apologize to maintain the relationship"
- DA: "Undskyld fordi det er det rigtige at gøre" / "Undskyld for at bevare forholdet"

NOT MATCH (different action despite same motivation):
- EN: "Call her to apologize" / "Write to her to apologize"
- DA: "Ring til hende for at undskylde" / "Skriv til hende for at undskylde"

IMPORTANT CONSTRAINTS

- Do NOT assume missing details.
- Do NOT use the dilemma to infer participants, intentions, or conditions.
- IGNORE motivations, reasons, or justifications (e.g., "for your own sake", "because X", "in order to Y").
- Differences in wording are acceptable ONLY if all explicit details match exactly.
- If in doubt, choose NOT MATCH.

## Examples
### Example 1
Example Dilemma:
Min nabo spiller høj musik hver aften efter kl. 22, og det forstyrrer min søvn. Jeg ved ikke, om jeg skal sige noget eller bare leve med det.

Example Solution A:
Bank på din nabos dør

Example Solution B:
Bank på naboens dør

Expected Output:
{{
  "rule_analysis": {{
    "core_action": "MATCH - Both solutions describe the same core action: knocking on door",
    "participants": "MATCH - Both refer to the same participant: neighbor (din nabo/naboens)",
    "details_and_modifiers": "MATCH - No additional details or modifiers present in either solution",
    "conditions": "MATCH - No conditions present in either solution",
    "action_order": "MATCH - Single action, no order to compare",
    "motivations": "N/A - No motivations stated in either solution"
  }},
  "is_match": true
}}

### Example 2
Example Dilemma:
Min teenage søn kommer konstant for sent hjem uden at give besked. Jeg er bekymret for hans sikkerhed, men vil ikke virke kontrollerende. Jeg ved ikke, hvordan jeg skal håndtere det.

Example Solution A:
Tal med din søn om at sende dig en besked, når han er forsinket

Example Solution B:
Tal med din søn om at komme hjem til tiden

Expected Output:
{{
  "rule_analysis": {{
    "core_action": "NOT MATCH - Solution A: talk about sending notification when delayed. Solution B: talk about coming home on time. These are fundamentally different actions",
    "participants": "MATCH - Both refer to the same participant: your son (din søn)",
    "details_and_modifiers": "NOT MATCH - Solution A includes 'når han er forsinket' (when delayed), Solution B includes 'til tiden' (on time)",
    "conditions": "MATCH - No conditions in either solution",
    "action_order": "MATCH - Single action in each",
    "motivations": "N/A - No motivations stated"
  }},
  "is_match": false
}}

### Example 3
Example Dilemma:
Min kollega låner ofte mine ting uden at spørge - kuglepenne, hæftemaskine, endda min kaffekrus. Det irriterer mig, men vi skal arbejde tæt sammen på et projekt. Jeg er i tvivl om, hvordan jeg skal tackle situationen uden at ødelægge samarbejdet.

Example Solution A:
Tag en privat samtale med din kollega og forklar høfligt, at du gerne vil spørges først, og hvis hun er forstående, så tilbyd at dele nogle af tingene fremover

Example Solution B:
Tag en privat samtale med din kollega og forklar høfligt, at du gerne vil spørges først

Expected Output:
{{
  "rule_analysis": {{
    "core_action": "PARTIAL MATCH - Both include 'have private conversation and explain politely', but Solution A has an additional conditional action (offer to share)",
    "participants": "MATCH - Both refer to the same participant: your colleague (din kollega)",
    "details_and_modifiers": "MATCH for shared parts - Both include 'privat' (private) and 'høfligt' (politely)",
    "conditions": "NOT MATCH - Solution A includes conditional 'hvis hun er forstående, så tilbyd at dele nogle af tingene' (if understanding, offer to share some things). Solution B has no conditional part",
    "action_order": "NOT MATCH - Solution A has sequence: conversation → conditional offer. Solution B only has conversation",
    "motivations": "N/A - No motivations stated"
  }},
  "is_match": false
}}

### Example 4
Example Dilemma:
Jeg er midt i 50'erne og sidder i toget på en varm dag iført en undertrøje og en t-shirt uden BH, fordi det er mest behageligt for mig. En kvinde overfor stirrede længe, rejste sig pludselig og sagde til mig, at det generer hende, at mine brystvorter stritter, og at jeg burde tage en trøje på.

Example Solution A:
Du skal klæde dig på for din egen skyld

Example Solution B:
Du skal klæde dig på for andres skyld

Expected Output:
{{
  "rule_analysis": {{
    "core_action": "MATCH - Both solutions describe the same core action: getting dressed (klæde dig på)",
    "participants": "MATCH - Both performed by the same participant: you (du)",
    "details_and_modifiers": "MATCH - No modifiers affecting WHAT or HOW to get dressed",
    "conditions": "MATCH - No conditions in either solution",
    "action_order": "MATCH - Single action",
    "motivations": "IGNORED - 'for din egen skyld' vs 'for andres skyld' are motivations/reasons and do not affect the action itself"
  }},
  "is_match": true
}}


## INPUT

Dilemma:
{dilemma}

Solution A:
{solution_a}

Solution B:
{solution_b}

TASK

Determine whether Solution A and Solution B describe the SAME fully specified solution according to the rules above.

OUTPUT FORMAT (STRICT)

Return ONLY the following JSON object, and nothing else:

{{
  "rule_analysis": {{
    "core_action": "MATCH/NOT MATCH - explanation",
    "participants": "MATCH/NOT MATCH - explanation",
    "details_and_modifiers": "MATCH/NOT MATCH - explanation",
    "conditions": "MATCH/NOT MATCH - explanation",
    "action_order": "MATCH/NOT MATCH - explanation",
    "motivations": "IGNORED/N/A - explanation"
  }},
  "is_match": bool
}}"""

        return prompt_template.format(
            dilemma=dilemma,
            solution_a=solution_a,
            solution_b=solution_b
        )

    @classmethod
    def prepare_prompts(cls, df, dilemma_colname, solutions_colname):

        prompts = []
        metadatas = []

        for i in tqdm(range(len(df))):

            for solution_dict_a, solution_dict_b in combinations(df.iloc[i][solutions_colname], 2):
                prep_prompt = cls.prepare_prompt(
                    dilemma=df.iloc[i][dilemma_colname],
                    solution_a=solution_dict_a['content'],
                    solution_b=solution_dict_b['content']
                )
                prompts.append(prep_prompt)
                metadatas.append(df.iloc[i].to_dict())

        return prompts, metadatas


def read_preds_from_dir(dirpath):
    data = []
    for filename in tqdm(os.listdir(dirpath), desc="Reading predictions"):
        if filename.endswith(".json"):
            data.append(
                json.load(open(os.path.join(dirpath, filename), "r"))
            )
    return pd.DataFrame(data)


def read_indexed_preds_from_dir_to_list(dirpath, expected_n):
    data = [None for _ in range(expected_n)]

    for filename in tqdm(os.listdir(dirpath), desc="Reading predictions"):
        if filename.endswith(".json"):
            idx = filename.split(".json")[0]
            idx = int(idx)

            with open(os.path.join(dirpath, filename), "r") as f:
                data[idx] = json.load(f)

    return data


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

        if "advised" not in data['parsed']:
            return True
        if len(data['parsed']["advised"]) == 0 and len(data['parsed']["not_advised"]) == 0:
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

            checkpoint_path = f"""Dilemma{metadict['dilemma_id']}.json"""
            checkpoint_path = os.path.join(save_dir, checkpoint_path)
            if self.check_if_needs_run(force_rewrite, checkpoint_path):
                indices_to_process.append(i)
                print(checkpoint_path)

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

        savefilename = f"""Dilemma{metadata['dilemma_id']}.json"""
        checkpoint_path = os.path.join(save_dir, savefilename)
        with open(checkpoint_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


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
        f"""Dilemma{metadict['dilemma_id']}.json""" for metadict in metadatas
    ]
    outputs = processor._load_all_results(
        task_savedir, expected_fienames
    )

    def check_if_ok(output):
        # print(output)
        if output["parsed"] is None or output["parsed"] == {}:
            return False
        if "advised" not in output["parsed"] or "not_advised" not in output["parsed"]:
            return False
        if not len(output["parsed"]["advised"]) and not len(output["parsed"]["not_advised"]):
            return False

        return True

    # Identify and retry failed parses
    for attempt in range(n_issue_resolve_attempts):
        issue_indices = [
            i for i, output in enumerate(outputs)
            if not check_if_ok(output)
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


def reformat_assign_ids_solutions(parsed_solutions):
    i = 0
    solutions = []

    for advise_type in parsed_solutions:
        for solution in parsed_solutions[advise_type]:
            solutions.append(
                {
                    "solution_id": f"{advise_type}-{i}",
                    "content": solution,
                    "advise_type": advise_type
                }
            )
            i += 1
    return solutions


def check_advised(postprocessed_solutions, solution):
    is_advised = False
    is_not_advised = False

    for solution_dict in postprocessed_solutions:
        if solution_dict['content'].strip() == solution.strip():
            if solution_dict['advise_type'] == 'advised':
                is_advised = True
            else:
                is_not_advised = True

            if is_advised and is_not_advised:
                return True, True

    return is_advised, is_not_advised


def load_extracted_data(loaddir, df, savecol):
    data = []
    # f"""Dilemma{metadict['dilemma_id']}.json"""
    for filename in os.listdir(loaddir):
        if filename.endswith(".json"):
            with open(os.path.join(loaddir, filename), "r") as f:
                loaded_json = json.load(f)
                if loaded_json['parsed']:
                    solutions_processed_dict = {
                        "extracted_solutions": loaded_json['parsed'],
                        "dilemma_id": loaded_json['metadata']['dilemma_id']
                    }
                    data.append(solutions_processed_dict)

    loaded_df = pd.DataFrame(data)

    # check for na
    loaded_df_nas = loaded_df[loaded_df[savecol].isna()]
    print("NA count (failed format/empty solutions): ", len(loaded_df_nas))
    if len(loaded_df_nas) > 0:
        print("You may want to rerun the script again to get rid of these issues")

    return pd.merge(
        df,
        loaded_df,
        how='left',
        left_on='dilemma_id',
        right_on='dilemma_id'
    )

if __name__ == "__main__":

    df = pd.read_json("preds_df.json")
    # preds_df.json should contain the following:
    # args:
    # - dilemma_colname: str, dilemma to provide to the model
    # - answer_colname: str, raw output from the model to extract solutions from
    # - dilemma_id: columns named answer_colname with the unique identifier of the dilemma


    # Settings
    BATCH_SIZE = 2_500
    N_FORMAT_ATTEMPTS = 1000

    savedir = "SoM-FinalData/extracted_postrpocessed"

    print("Num samples to run:", len(df))

    # Initialize processors
    extraction_postprocessing_processor = VLLMBatchProcessor(
        model_name="openai/gpt-oss-120b",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        pipeline_parallel_size=1,
        # max_model_len=30_000
        max_num_seqs=2056

    )


    extract_prompts, metadata = ExtractionPrompter.prepare_prompts(
        df=df,
        dilemma_colname="first_person_dilemma",
        answer_colname="odin-large -- pred",
    )

    run_parallel_issue_fixing(
        processor=extraction_postprocessing_processor,
        prompts=extract_prompts,
        savedir=savedir,
        taskname=ExtractionPrompter.taskname,
        batch_size=BATCH_SIZE,
        response_format=FinalSolutions,
        parse_function=parse_extraction,
        n_issue_resolve_attempts=N_FORMAT_ATTEMPTS,
        metadatas=metadata
    )

    # load data to the df from the dir
    df = load_extracted_data(
        loaddir=os.path.join(savedir, ExtractionPrompter.taskname),
        df=df,
        savecol="extracted_solutions"
    )


    postprocess_prompts, metadata = SolutionPostprocessingPrompter.prepare_prompts(
        df=df,
        dilemma_colname="first_person_dilemma",
        solutions_colname="extracted_solutions",
        answer_colname="odin-large -- pred",
    )


    run_parallel_issue_fixing(
        processor=extraction_postprocessing_processor,
        prompts=postprocess_prompts,
        savedir=savedir,
        taskname=SolutionPostprocessingPrompter.taskname,
        batch_size=BATCH_SIZE,
        response_format=FinalSolutions,
        parse_function=parse_extraction,
        n_issue_resolve_attempts=N_FORMAT_ATTEMPTS,
        metadatas=metadata
    )
    print("Done Processing!")

    df = load_extracted_data(
        loaddir=os.path.join(savedir, ExtractionPrompter.taskname),
        df=df,
        savecol="odin-large -- postprocessed_extracted_solutions_first_person"
    )
    df.to_json("processed_extracted_solutions_df.json")

