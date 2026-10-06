"""Optional local/API providers behind small embedding and decision contracts.

The HTTP adapters use only the standard library. Models are never selected or
downloaded implicitly. A decision backend returns P(true), not generated text.
"""

import json
import math
import os
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from . import _decision_tasks
from .jev import JevScorer


def _positive(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _model(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("An explicit model name or local model path is required")
    return value


def _probabilities(values, questions):
    if not isinstance(values, dict) or set(values) != set(questions):
        raise ValueError("Decision response must have exactly the requested question IDs")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError("Decisions must be finite P(true) numbers in [0, 1], not logits or text")
    return {key: float(values[key]) for key in questions}


class _HTTP:
    def __init__(self, endpoint, *, api_key=None, api_key_env=None, timeout=60, max_requests=1000):
        parsed = urlsplit(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.fragment:
            raise ValueError("Provide an http(s) endpoint without embedded credentials or fragments")
        if api_key is not None and api_key_env is not None:
            raise ValueError("Supply api_key or api_key_env, not both")
        if api_key_env is not None:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"Set {api_key_env} before using this provider")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        _positive("max_requests", max_requests)
        self.endpoint, self._key = endpoint, api_key
        self.timeout, self.max_requests, self.requests = timeout, max_requests, 0
        self._lock = Lock()

    def post(self, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = "Bearer " + self._key
        with self._lock:
            if self.requests >= self.max_requests:
                raise RuntimeError("Provider request limit reached; increase max_requests explicitly")
            self.requests += 1  # Failed requests also consume the allowance; no hidden retries.
        try:
            with urlopen(Request(self.endpoint, data, headers, method="POST"), timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"Provider HTTP {error.code}; check model, context limits, credentials and service") from None
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"Provider connection failed ({type(error).__name__}); check the endpoint and server") from None
        except (ValueError, UnicodeError):
            raise ValueError("Provider returned invalid JSON") from None
        if not isinstance(result, dict) or "error" in result:
            raise ValueError("Provider did not return a successful response object")
        return result


class _Embeddings:
    def __init__(self, model, batch_size=64):
        self.model = _model(model)
        _positive("batch_size", batch_size)
        self.batch_size = batch_size
        self._cache = {}
        self._dimension = None

    def embed(self, texts, purpose):
        texts = list(texts)
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Embedding inputs must be nonempty strings")
        missing = list(dict.fromkeys(t for t in texts if (purpose, t) not in self._cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            rows = list(self._encode(batch, purpose))
            if len(rows) != len(batch):
                raise ValueError("Embedding response must contain one vector per input")
            vectors = []
            for row in rows:
                vector = tuple(float(v) for v in row)
                norm = math.hypot(*vector)
                if not vector or not all(math.isfinite(v) for v in vector) or not math.isfinite(norm) or norm == 0:
                    raise ValueError("Embedding vectors must be finite and nonzero")
                dimension = self._dimension or len(vectors[0] if vectors else vector)
                if len(vector) != dimension:
                    raise ValueError("Embedding dimensions changed within the provider")
                vectors.append(vector)
            if vectors:
                self._dimension = len(vectors[0])
            self._cache.update(((purpose, text), vector) for text, vector in zip(batch, vectors))
        return [list(self._cache[purpose, text]) for text in texts]


class EmbeddingAPI(_Embeddings):
    """OpenAI-format /embeddings or Ollama /api/embed, hosted or localhost."""

    def __init__(self, model, *, endpoint, format="openai", batch_size=64, **http):
        super().__init__(model, batch_size)
        if format not in ("openai", "ollama"):
            raise ValueError("Embedding format must be openai or ollama")
        self.format = format
        self.http = _HTTP(endpoint, **http)

    def _encode(self, texts, purpose):
        body = {"model": self.model, "input": texts}
        if self.format == "ollama":
            body["truncate"] = False
        else:
            body["encoding_format"] = "float"
        result = self.http.post(body)
        try:
            if self.format == "ollama":
                return result["embeddings"]
            rows = result["data"]
            if (not isinstance(rows, list) or len(rows) != len(texts)
                    or any(type(row.get("index")) is not int for row in rows)
                    or {row["index"] for row in rows} != set(range(len(texts)))):
                raise ValueError("Embedding response has missing, duplicate or invalid input indices")
            return [row["embedding"] for row in sorted(rows, key=lambda row: row["index"])]
        except (KeyError, TypeError):
            raise ValueError("Embedding response does not match the configured API format") from None


class OllamaEmbeddings(EmbeddingAPI):
    def __init__(self, model, *, base_url="http://localhost:11434", **kwargs):
        super().__init__(model, endpoint=base_url.rstrip("/") + "/api/embed", format="ollama", **kwargs)


class CallableEmbeddings(_Embeddings):
    """Adapt any local model or SDK: encode(texts, purpose) -> vectors."""

    def __init__(self, encode, *, model, batch_size=64):
        super().__init__(model, batch_size)
        if not callable(encode):
            raise ValueError("encode must be callable")
        self._encode = encode


class SentenceTransformerEmbeddings(_Embeddings):
    """Optional in-process model; local_files_only=True forbids downloads."""

    def __init__(self, model, *, device=None, local_files_only=False, batch_size=32):
        super().__init__(model, batch_size)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError('Install local embedding support from the checkout: python -m pip install ".[local]"') from None
        self.encoder = SentenceTransformer(model, device=device, local_files_only=local_files_only,
                                           trust_remote_code=False)

    def _encode(self, texts, purpose):
        tokenizer = getattr(self.encoder, "tokenizer", None)
        limit = getattr(self.encoder, "max_seq_length", None)
        if callable(tokenizer) and isinstance(limit, int):
            tokens = tokenizer(texts, truncation=False, padding=False)["input_ids"]
            if any(len(row) > limit for row in tokens):
                raise ValueError("Embedding input exceeds the local model context; choose a longer-context model or explicit custom adapter")
        return self.encoder.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                   show_progress_bar=False, prompt="").tolist()


class SystemOneAPI:
    """Jev-compatible state/questions/answers protocol with any model name."""

    def __init__(self, model, *, endpoint, max_request_bytes=65536, max_questions=64, **http):
        self.model = _model(model)
        _positive("max_request_bytes", max_request_bytes)
        _positive("max_questions", max_questions)
        self.max_request_bytes = max_request_bytes
        self.max_questions = max_questions
        self.http = _HTTP(endpoint, **http)

    def decide(self, state, questions):
        # Pack the full wire body, including source text in structured questions.
        # Preflight all single questions before making calls; never truncate text.
        def payload(items):
            return {"model": self.model, "state": state, "questions": dict(items)}
        def fits(items):
            return (len(items) <= self.max_questions and
                    len(json.dumps(payload(items), ensure_ascii=False, allow_nan=False).encode()) <= self.max_request_bytes)
        batches, batch = [], []
        for item in questions.items():
            if not fits([item]):
                raise ValueError("One decision exceeds max_request_bytes with complete context; reduce source/selection allowances or use a larger-context backend")
            if batch and not fits(batch + [item]):
                batches.append(batch)
                batch = []
            batch.append(item)
        if batch:
            batches.append(batch)
        values = {}
        for items in batches:
            response = self.http.post(payload(items))
            try:
                answers = response["answers"]
                if set(answers) != {key for key, _ in items} or any(a["type"] != "noul" for a in answers.values()):
                    raise ValueError("Decision response has incorrect answer IDs or types")
                values.update(_probabilities({key: answer["noul"] for key, answer in answers.items()}, dict(items)))
            except (KeyError, TypeError):
                raise ValueError("Expected System One noul answers for every question") from None
        return values


class OllamaDecisions(SystemOneAPI):
    def __init__(self, model, *, base_url="http://localhost:11434", **kwargs):
        super().__init__(model, endpoint=base_url.rstrip("/") + "/v1/systemone", **kwargs)


class JevAPI(SystemOneAPI):
    def __init__(self, model="jev-1.13.0", *, provider="typesafe", api_key=None, api_key_env=None, **kwargs):
        providers = {"typesafe": ("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY"),
                     "openrouter": ("https://openrouter.ai/api/alpha/decisions", "OPENROUTER_API_KEY")}
        if provider not in providers:
            raise ValueError("Jev provider must be typesafe or openrouter; use SystemOneAPI for custom endpoints")
        endpoint, default_env = providers[provider]
        if provider == "openrouter" and model == "jev-1.13.0":
            model = "typesafe/jev-1.13"
        super().__init__(model, endpoint=endpoint, api_key=api_key,
                         api_key_env=api_key_env if api_key_env is not None else default_env if api_key is None else None,
                         **kwargs)


class CallableDecisions:
    """Adapt any local model/SDK: decide(state, questions) -> {id: P(true)}."""

    def __init__(self, decide, *, model, max_requests=1000):
        self.model = _model(model)
        if not callable(decide):
            raise ValueError("decide must be callable")
        _positive("max_requests", max_requests)
        self.callback, self.max_requests, self.requests = decide, max_requests, 0
        self._lock = Lock()

    def decide(self, state, questions):
        with self._lock:
            if self.requests >= self.max_requests:
                raise RuntimeError("Local decision call limit reached")
            self.requests += 1
        return _probabilities(self.callback(state, questions), questions)


class DecisionModel:
    """Use one backend for splitting, central sentences and all retrieval stages.

    Routing candidates and paragraph pairs remain question-local. Shared-set
    selection intentionally sees the complete shared packet in both orders.
    """

    score_kind = "probability"
    _batch = JevScorer._batch
    _representative_payload = staticmethod(JevScorer._representative_payload)
    score_many = JevScorer.score_many
    representatives = JevScorer.representatives
    route_content = _decision_tasks.route_content
    compare = _decision_tasks.compare
    score_pool = _decision_tasks.score_pool

    def __init__(self, backend, *, batch_size=32, max_state_chars=250000, max_concurrency=1):
        if not callable(getattr(backend, "decide", None)):
            raise ValueError("Backend must implement decide(state, questions)")
        self.model = _model(backend.model)
        for name, value in (("batch_size", batch_size), ("max_state_chars", max_state_chars), ("max_concurrency", max_concurrency)):
            _positive(name, value)
            setattr(self, name, value)
        self.backend, self._cache = backend, {}
        self.name = "decision:" + self.model

    def _request(self, state, questions):
        if "nodes" in state:
            scoped = {}
            for key, question in questions.items():
                reference = "nodes.n" + key[1:]
                scoped[key] = {**question, "instructions": {
                    "target": state["nodes"]["n" + key[1:]],
                    "decision": question["instructions"].replace(reference, "`target`")}}
            state, questions = {k: v for k, v in state.items() if k != "nodes"}, scoped
        elif "pairs" in state and "question" in state:
            questions = {key: {**q, "instructions": {**state["pairs"]["p" + key[1:]],
                         "decision": q["instructions"]}} for key, q in questions.items()}
            state = {"question": state["question"]}
        return _probabilities(self.backend.decide(state, questions), questions)

    def score(self, left, right):
        return self.score_many([(left, right)])[0]

    def representative(self, sentence, context):
        return self.representatives([(sentence, context)])[0]

    def metadata(self):
        return {"model": self.model, "adapter": type(self.backend).__name__, "score_kind": self.score_kind}
