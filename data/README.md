# Public input data

## Refusal-direction corpus

`refusal_direction/` contains fixed train and validation splits of harmful and harmless requests. `PROVENANCE.json` records the source and split construction. H1 fits a difference-in-means direction on 64 harmful and 64 harmless training requests and evaluates it on disjoint sets of 32 requests per class.

## Paired jailbreak dataset

`jailbreaks/jailbreak_dataset.json` contains 50 harmful goals for each of seven transformations, paired with a matched control that preserves the surrounding framing while removing the active jailbreak instruction where possible.

The classes are:

- **Authoritative tone:** presents the request as if permission or expert authority has already been established.
- **Fictional framing:** embeds the request in a fictional, hypothetical, or role-based setting.
- **Narrative sandwich:** places the harmful request between longer benign-looking passages.
- **Prefill:** inserts a model-specific compliant opening into the assistant turn before generation.
- **Roleplay:** assigns the model a persona expected to answer the request.
- **Prompted prefill, long:** uses detailed user-side rules that ban refusal language, require a direct answer, and specify an opening.
- **Prompted prefill, short:** uses a brief user-side instruction not to refuse or add disclaimers.

The public dataset is an experimental input and contains harmful requests. It does not contain generated model responses. Class membership describes the prompt transformation and does not imply that a particular generation succeeded.
