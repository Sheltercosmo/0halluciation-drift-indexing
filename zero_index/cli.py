import argparse
import json
from pathlib import Path

from .index import DocumentIndex, build_index
from .jev import JevScorer
from .model import Config, LexicalJaccard
from .ranking import find
from .structure import HeadingHint


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and inspect a source-backed content tree")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("input", type=Path)
    build.add_argument("-o", "--output", type=Path, required=True)
    build.add_argument("--scorer", choices=("lexical", "jev"), default="lexical")
    build.add_argument("--prior", type=float, default=0.7)
    build.add_argument("--reference-prior", type=float, default=0.5)
    build.add_argument("--cutoff", type=float, default=0.5)
    build.add_argument("--drop", type=float, default=0.2)
    build.add_argument("--sentence-budget", type=int, default=0, help="0 searches every sentence (default); otherwise >= 2 per node")
    build.add_argument("--batch-size", type=int, default=64, help="Maximum parallel Jev questions per request")
    build.add_argument("--max-concurrency", type=int, default=1, help="Maximum in-flight Jev HTTP requests")
    build.add_argument("--max-calls", type=int, default=1000)
    build.add_argument("--model", help="Defaults to the pinned model for the selected provider")
    build.add_argument("--provider", choices=("typesafe", "openrouter"), default="openrouter")
    build.add_argument("--headings-json", type=Path, help="Optional source-aligned heading hints from a document parser")
    search = commands.add_parser("find", help="Propose needed content and rerank sentence nodes")
    search.add_argument("index", type=Path)
    search.add_argument("need")
    search.add_argument("--question", default="")
    search.add_argument("--reranker", choices=("lexical", "jev"), default="lexical")
    search.add_argument("--scope", default="root")
    search.add_argument("--candidates", type=int, default=64, help="0 sends all scoped leaves to the reranker")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--batch-size", type=int, default=64)
    search.add_argument("--max-concurrency", type=int, default=1, help="Maximum in-flight Jev HTTP requests")
    search.add_argument("--provider", choices=("typesafe", "openrouter"), default="openrouter")
    search.add_argument("--model")
    search.add_argument("--max-calls", type=int, default=1000)
    for name in ("children", "read", "trace", "outline", "parent"):
        command = commands.add_parser(name)
        command.add_argument("index", type=Path)
        if name not in ("trace", "outline"):
            command.add_argument("node_id", nargs="?", default="root")
    args = parser.parse_args()
    try:
        if args.command == "build":
            if args.input.resolve() == args.output.resolve():
                raise ValueError("Output must differ from input")
            config = Config(same_topic_prior=args.prior, posterior_cutoff=args.cutoff,
                            minimum_drop=args.drop, probability_reference_prior=args.reference_prior,
                            sentence_budget=None if args.sentence_budget == 0 else args.sentence_budget)
            scorer = (JevScorer(model=args.model, max_calls=args.max_calls, batch_size=args.batch_size,
                                provider=args.provider, max_concurrency=args.max_concurrency)
                      if args.scorer == "jev" else LexicalJaccard())
            hints = None
            if args.headings_json:
                hints = [HeadingHint(**item) for item in json.loads(args.headings_json.read_text(encoding="utf-8"))]
            # newline='' preserves CRLF so citations address the original decoded input.
            with args.input.open(encoding="utf-8", newline="") as stream:
                index = build_index(stream.read(), source_name=args.input.name, scorer=scorer, config=config,
                                    heading_hints=hints)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote {args.output} ({len(list(index.root.walk()))} nodes; {scorer.name})")
            return
        index = DocumentIndex.from_dict(json.loads(args.index.read_text(encoding="utf-8")))
        if args.command == "find":
            reranker = (JevScorer(batch_size=args.batch_size, provider=args.provider, model=args.model,
                                 max_calls=args.max_calls, max_concurrency=args.max_concurrency)
                        if args.reranker == "jev" else None)
            result = find(index, args.question or args.need, args.need, reranker=reranker, scope_id=args.scope,
                          candidate_limit=None if args.candidates == 0 else args.candidates, top_k=args.top_k)
        elif args.command == "outline":
            result = index.outline()
        else:
            result = index.decisions if args.command == "trace" else getattr(index, args.command)(args.node_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
