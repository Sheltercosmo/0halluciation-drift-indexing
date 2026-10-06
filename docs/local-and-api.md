# Install and use local or API models

Python 3.10 or newer. The base package and HTTP clients use only the Python standard library. Model runtimes are optional. The installed package contains the complete EEJ pipeline; it does not depend on the repository's experiment scripts.

## Install locally

```sh
git clone https://github.com/Sheltercosmo/0halluciation-drift-indexing.git
cd 0halluciation-drift-indexing
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```sh
python -m pip install .
zero-index --help
```

Use `python -m pip install -e .` instead when developing this repository. Every `zero-index` command can also be invoked as `python -m zero_index`.

## Fully local quick start

Install and start [Ollama](https://ollama.com/download), version 0.35.0 or later for its local decision endpoint. Download an embedding model and a decision model once:

```sh
ollama pull embeddinggemma
ollama pull nimble
zero-index init -o local.json
zero-index index examples/structured.md --config local.json -o document.index.json
zero-index search document.index.json "What are the visitor opening times?" --config local.json
```

The defaults use local `http://localhost:11434/api/embed` and `/v1/systemone`; no API key is needed. Model downloads need connectivity, while inference runs on the local server. If the server is not already running, start `ollama serve` in another terminal. See Ollama's [embedding API](https://docs.ollama.com/api/embed) and [decision setup](https://docs.ollama.com/capabilities/decision).

Replace `examples/structured.md` with a Markdown or plain-text file. The index preserves the source text, native headings, paragraph IDs and character offsets. Search prints JSON containing `paragraphs` with complete evidence text and source spans, along with the effective configuration and decision traces. It does not generate an answer.

```sh
# Evidence needs can come from your application or an LLM.
zero-index search document.index.json "What changed?" --need "Measurements before and after the change" --config local.json

# Independent direct embedding retrieval plus the same Jev evidence selector.
zero-index search document.index.json "What changed?" --hybrid --config local.json -o evidence.json

# Optional smaller effort settings; init never overwrites an existing config.
zero-index init -o low-effort.json --effort low
```

Edit `local.json` to choose models, endpoints, request limits and [retrieval hyperparameters](retrieval-standard.md). EEJ is the default: embeddings for splitting and central sentences, decisions for search and selection. `--variant JJJ` uses decision-model indexing instead. Rebuild the index after changing indexing settings or models. Search settings can change without rebuilding.

## Mix local models and hosted APIs

Embedding and decision providers are independent. Keep local embeddings and replace only `decision` in the generated JSON to use hosted Jev:

```json
"decision": {
  "provider": "typesafe",
  "model": "jev-1.13.0",
  "api_key_env": "TYPESAFE_API_KEY",
  "max_requests": 1000
}
```

Set the key in your shell: `export TYPESAFE_API_KEY="..."` on macOS/Linux, or `$env:TYPESAFE_API_KEY="..."` in PowerShell. Configuration stores the environment variable name, not the credential. Native Jev uses the documented [System One request/response protocol](https://docs.typesafe.ai/api). Use a model ID supported by your provider; the measured historical run used `jev-1.13.0`.

An embedding API with the standard `/embeddings` JSON shape can replace the local embedding provider:

```json
"embedding": {
  "provider": "openai",
  "model": "YOUR_EMBEDDING_MODEL",
  "endpoint": "https://YOUR_SERVICE/v1/embeddings",
  "api_key_env": "EMBEDDING_API_KEY",
  "batch_size": 64,
  "max_requests": 1000
}
```

Here `openai` names the wire format, not a required vendor. A compatible local server can use `http://localhost:8000/v1/embeddings` without `api_key_env`. Each response must contain `data` entries with `index` and numeric `embedding`; the adapter restores input order and checks dimensions. For other SDK/API formats, use the Python callback adapter below.

| Provider setting | Execution | Required fields |
| --- | --- | --- |
| Embedding `ollama` | Local or explicitly configured server | `model`; optional `base_url` |
| Embedding `openai` | Compatible local or hosted HTTP API | `model`, complete `endpoint`; optional `api_key_env` |
| Embedding `sentence-transformers` | In-process local model | `model` path or model ID; optional `device`, `local_files_only` |
| Decision `ollama` | Local System One model | `model`; optional `base_url` |
| Decision `typesafe` | Hosted Jev | `model`; defaults to `TYPESAFE_API_KEY` |
| Decision `openrouter` | Hosted decision API | Namespaced `model`, e.g. `typesafe/jev-1.13`; defaults to `OPENROUTER_API_KEY` |
| Decision `systemone` | Any compatible local or hosted decision server | `model`, complete `endpoint`; optional `api_key_env` |

`systemone` accepts arbitrary model names and expects `state` + named `questions`, returning matching `answers` with `{ "type": "noul", "noul": 0.8 }`. It does not assume a chat-completion endpoint or convert generated prose into probabilities. A compatible scoring model is required; an arbitrary chat model cannot be substituted solely by changing its name.

## Python usage

```python
from zero_index import RetrievalAdapter, load_document
from zero_index.providers import OllamaEmbeddings, OllamaDecisions

retriever = RetrievalAdapter.from_models(
    OllamaEmbeddings("embeddinggemma"),
    OllamaDecisions("nimble"),
)
document = load_document("document.md")
index = retriever.build_index(document)
result = retriever.retrieve(document, index, "What evidence supports the finding?")

for paragraph in result["paragraphs"]:
    print(paragraph["node_id"], paragraph["text"])
```

Replace `OllamaDecisions(...)` with `JevAPI(api_key_env="TYPESAFE_API_KEY")` for hosted Jev, or `SystemOneAPI(model="your-model", endpoint="http://localhost:9000/v1/systemone")` for another server. All are in `zero_index.providers`.

Pass `config=RetrievalConfig.for_effort("low")` or your edited configuration to `from_models`. Explicit configurations are honored exactly. The convenience setup and generated local configuration use an empty embedding prefix. If your model needs task instructions, supply them in the embedding callback or configure `indexing.paragraph_embedding_prefix`; the historical evaluation prefix is specific to that setup.

### Sentence Transformers in process

```sh
python -m pip install ".[local]"
```

```python
from zero_index.providers import SentenceTransformerEmbeddings

embedding = SentenceTransformerEmbeddings(
    "/path/to/your/local/embedding-model",
    local_files_only=True,
    device="cpu",  # Or a device supported by your installed runtime.
)
```

Use this object in `from_models` alongside any decision backend. `local_files_only=True` prevents model downloads; a model ID with `False` permits the model library to fetch its weights. Model-specific prompts or unusual encoders can use the callback adapter. See the [Sentence Transformers API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).

### Any local model or SDK through callbacks

```python
from zero_index import RetrievalAdapter
from zero_index.providers import CallableEmbeddings, CallableDecisions

# Implement these with your own installed model or SDK:
# encode_batch(texts, purpose) -> one numeric vector per text, in input order
# decide_batch(state, questions) -> {question_id: float_probability_true}
embedding = CallableEmbeddings(encode_batch, model="my-embedding/version")
decision = CallableDecisions(decide_batch, model="my-decision/version")
retriever = RetrievalAdapter.from_models(embedding, decision)
```

The embedding callback receives `splitting`, `central-sentences`, `retrieval-document`, or `retrieval-query` as its purpose. Use it to apply your model's document/query instructions. Decision callbacks receive the full structured state, question instructions and criteria. Return exactly the requested IDs with finite numbers in `[0, 1]`; convert your model's logits or output schema explicitly. Replacing the model does not establish equivalence to Jev's measured accuracy or calibration.

For a backend with separate task APIs, the lower-level `RetrievalAdapter(route=..., compare=..., select=..., embed=..., embedding_model=..., jev=...)` remains available. `route(question, need, cards)` returns one score per card; `compare(question, pairs)` returns one preference probability per ordered pair; `select(question, packet, target_ids)` returns one score per target. J indexing additionally needs a scorer implementing `score`/`score_many` and representative scoring. No embedding or decision vendor is hard-coded into these contracts.

## Limits and verification

Provider `max_requests` limits HTTP attempts per process and provider, including failed attempts; there are no automatic retries. `decision_batching` controls questions per batch, concurrency and the state-size guard. System One also packs at most `max_questions` per request (default 64). Model and request limits are independent of the [search decision ceiling](retrieval-standard.md#search-effort); they do not represent a dollar budget.

Ollama embeddings use `truncate=false`. System One requests are packed by complete UTF-8 body size, including source text in question instructions. An oversized single decision fails without silently dropping text. Its default 64 KiB request cap follows the [local endpoint contract](https://docs.ollama.com/api/systemone); a larger-context backend can use an explicit `max_request_bytes`. The backend also enforces its token context window. For long documents, adjust shared targets/source cues or use a model that can hold the required evidence.

The wheel smoke test installs and runs outside the repository, indexes a document, performs hybrid retrieval through local HTTP protocol fixtures, and checks exact returned spans. Provider tests check scope parity with the retained Jev prompts. These validate integration, not the accuracy of a new local model; historical scores remain tied to their recorded configuration.
