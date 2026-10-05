# Data attribution

**QASPER v0.3 test:** Pradeep Dasigi and colleagues, [A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers](https://aclanthology.org/2021.naacl-main.365/). The [AllenAI dataset card](https://huggingface.co/datasets/allenai/qasper) lists CC BY 4.0. The run uses the official test archive and its supplied evaluator.

**QuALITY v1.0.1 HTML-stripped development:** Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi and colleagues, [QuALITY: Question Answering with Long Input Texts, Yes!](https://github.com/nyu-mll/quality). Each released article has its own license field. The 115 articles used here comprise 86 Project Gutenberg texts, 19 Open American National Corpus texts and 10 texts marked CC BY 4.0 in the release.

Exact URLs, revisions and SHA-256 checksums are in [the source manifest](../frontier/sources.json); the [registered selection](selection.json) identifies every question and source document. Published predictions and offsets do not redistribute the source articles or scientific papers. Download the upstream releases to reconstruct the inputs.

Run the following command from the repository root to fetch just these two inputs (about 13 MB), without downloading the larger research inventory or making model calls:

~~~sh
python scripts/fetch_bounded_data.py
~~~
