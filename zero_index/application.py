"""JSON provider configuration and installed index/search commands."""

from dataclasses import replace
import json
from pathlib import Path

from .configuration import RetrievalConfig
from .documents import load_document
from .index import DocumentIndex
from .providers import (EmbeddingAPI, OllamaEmbeddings, SentenceTransformerEmbeddings,
                        SystemOneAPI, OllamaDecisions, JevAPI, DecisionModel)
from .standard import RetrievalAdapter


def default_application():
    config = RetrievalConfig()
    config = replace(config, indexing=replace(config.indexing, paragraph_embedding_prefix=""))
    return {"schema_version": 1, "retrieval": config.to_dict(),
            "embedding": {"provider": "ollama", "model": "embeddinggemma", "max_requests": 1000},
            "decision": {"provider": "ollama", "model": "nimble", "max_requests": 1000},
            "decision_batching": {"batch_size": 32, "max_state_chars": 250000, "max_concurrency": 1}}


def adapter_from_config(value, *, indexing_only=False):
    if not isinstance(value, dict) or set(value) - {"schema_version", "retrieval", "embedding", "decision", "decision_batching"}:
        raise ValueError("Unknown application configuration fields")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported application schema_version")
    config = RetrievalConfig.from_dict(value.get("retrieval", {}))
    def provider(key, factories):
        data = value.get(key)
        if not isinstance(data, dict):
            raise ValueError(f"A {key} provider configuration is required")
        data = dict(data)
        kind = data.pop("provider", None)
        if "api_key" in data:
            raise ValueError("Use api_key_env in JSON configuration; pass api_key directly only in Python")
        if kind not in factories:
            raise ValueError(f"Unsupported {key} provider: {kind}")
        return factories[kind](**data)
    try:
        embeddings = None
        if value.get("embedding") is not None:
            embeddings = provider("embedding", {
                "ollama": OllamaEmbeddings, "openai": EmbeddingAPI,
                "sentence-transformers": SentenceTransformerEmbeddings})
        if "E" in config.variant[:2] and embeddings is None:
            raise ValueError("EE/EJ/JE indexing requires an embedding provider")
        if indexing_only and "J" not in config.variant[:2]:
            # EE indexing does not require a Jev key or a running decision model.
            def unused(*args):
                raise RuntimeError("Load retrieval providers before searching")
            return RetrievalAdapter(unused, unused, unused, config, embeddings.embed, embeddings.model)
        decisions = provider("decision", {
            "ollama": OllamaDecisions, "systemone": SystemOneAPI,
            "typesafe": JevAPI,
            "openrouter": lambda **kwargs: JevAPI(provider="openrouter", **kwargs)})
        model = DecisionModel(decisions, **value.get("decision_batching", {}))
        return RetrievalAdapter.from_models(embeddings, model, config=config)
    except TypeError as error:
        raise ValueError(f"Invalid provider configuration: {error}") from None


def register_commands(commands):
    init = commands.add_parser("init", help="Write an editable local/API provider configuration")
    init.add_argument("-o", "--output", type=Path, required=True)
    init.add_argument("--effort", choices=("low", "standard", "high"), default="standard")
    init.add_argument("--variant", choices=("EEJ", "EJJ", "JEJ", "JJJ"), default="EEJ")
    for name, help_text in (("index", "Index Markdown/plain text with configured providers"),
                            ("search", "Retrieve whole evidence paragraphs from a saved index")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("input", type=Path)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("-o", "--output", type=Path, required=name == "index")
        if name == "search":
            command.add_argument("question")
            command.add_argument("--need", action="append", default=[], help="Evidence need from your application or LLM; repeatable")
            command.add_argument("--hybrid", action="store_true", help="Add independent direct embedding retrieval")


def run_command(args):
    if args.command == "init":
        if args.output.exists():
            raise ValueError("Configuration exists; edit it or choose a new output path")
        value = default_application()
        config = RetrievalConfig.for_effort(args.effort, variant=args.variant)
        config = replace(config, indexing=replace(config.indexing, paragraph_embedding_prefix=""))
        value["retrieval"] = config.to_dict()
        if args.variant == "JJJ":
            value["embedding"] = None
    else:
        if args.output and args.output.resolve() in (args.input.resolve(), args.config.resolve()):
            raise ValueError("Output must differ from input and provider configuration")
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        if args.command == "index":
            adapter = adapter_from_config(settings, indexing_only=True)
            doc = load_document(args.input)
            index = adapter.build_index(doc)
            value = {"schema_version": 1, "document": doc, "index": index.to_dict(),
                     "retrieval_config": adapter.config.to_dict()}
        else:
            bundle = json.loads(args.input.read_text(encoding="utf-8"))
            if bundle.get("schema_version") != 1:
                raise ValueError("Unsupported document bundle schema")
            doc, index = bundle["document"], DocumentIndex.from_dict(bundle["index"])
            adapter = adapter_from_config(settings)
            dense = adapter.direct_embedding_ranking(doc, args.question, args.need) if args.hybrid else None
            value = adapter.retrieve(doc, index, args.question, args.need, dense_ranking=dense)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text, end="")
