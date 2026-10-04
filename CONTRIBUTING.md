# Contributing

Start with the [algorithm](docs/algorithm.md) and [retrieval contract](docs/retrieval.md). The runtime uses only the Python standard library. Python 3.10 or newer is required.

```sh
python -m unittest discover -s tests -v
python -m examples.bottom_up
python -m evals.run --replay evals/results/pilot-v1-live --output output/my-replay
```

Keep source spans exact, preserve explicit call/read budgets, and fail visibly when a provider response is invalid. Keep topic similarity, representative selection and query relevance as separate judgments. Add regression coverage when those behaviors change.

Each live archive includes its frozen evaluator, runtime and corpus. Replay verifies their original hashes and loads that snapshot, so later code changes cannot silently alter historical results. Create a separately versioned protocol/run for new experiments. Do not regenerate a frozen manifest to make a changed implementation pass. The v1 replay allows at most `1e-12` roundoff in displayed relevance scores; ranks, evidence and aggregate metrics must match exactly.

Live evaluations require a TypeSafe key in `TYPESAFE_API_KEY` and incur provider usage. Use a new output directory for every attempt. Archive failures alongside successes. Never commit keys, environment files, authorization headers, or private source documents.

The [topic manifest](.github/topics.json) contains suggested GitHub repository topics. It does not apply them to a remote automatically. After a remote exists, a maintainer can apply those topics in the repository settings. Add citations for research claims and distinguish measured behavior from proposals.
