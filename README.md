# Austronesian Numeral Morphological Analysis by LLMs

Code and selected model outputs for a study of numeral-form analysis in Austronesian languages.

## Data and provenance

The experiment's numeral word forms were collected from Mark Rosenfelder's [The Numbers List](https://www.zompist.com/numbers.shtml). Its [notes](https://www.zompist.com/numnotes.html) acknowledge Eugene S. L. Chan for supplying an Austronesian database.

The source pages reviewed for this release do not state a license or grant permission to redistribute the compiled data. To avoid republishing it without permission, this repository does not include the source TSV files. Attribution alone should not be treated as permission. The pipeline expects a locally available `data/gold_sheet.tsv`; obtain and use a copy only under terms that permit your use.

The included model result files contain model responses and gold formulas, but not the source numeral word forms.

## Experiments

Five open-weight models were evaluated in two conditions using the neutral prompt variant:

- `zeroshot_noname`: no worked example; language name withheld.
- `oneshot_noname`: one Seediq worked example; language name withheld.

The `results/` directory contains the JSONL output and run metadata for these ten model-condition runs. Intermediate logs and candidate-review files are not included.

## Completeness note

The Llama 3.1 8B one-shot run has no valid model response for language ID 529 (Unank, North Watut), so its reported statistics use the other 898 records (8,701 scored cells). The reported accuracy remains 56.2%. The other nine model-condition runs contain 899 records each. The missing API response does not affect the included calculations; the paper reports this exception explicitly.

## Running

The scripts require Python and a local, lawfully obtained `data/gold_sheet.tsv`. API-backed inference also requires the relevant provider key in an environment variable; do not commit keys or `.env` files.

```powershell
python pipeline.py --model llama-3.1-8b-instant --condition zeroshot_noname --prompt-variant neutral
python paper_numbers.py
```

`paper_numbers.py` regenerates the paper statistics from the local gold sheet and the included model outputs. Because the source TSV is not distributed here, those calculations require the user to supply it locally.

## Source code license

The original project describes its code as MIT-licensed. This license applies to the code only; it does not grant rights to The Numbers List or other third-party data.
