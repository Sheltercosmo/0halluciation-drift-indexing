"""Source-backed content trees and provider-independent navigation tools."""

from bisect import bisect_right
from dataclasses import asdict, dataclass, field
import hashlib
from typing import Iterator

from .model import Config, LexicalJaccard, Similarity
from .parse import Block, sentence_spans
from .segment import central_sentences, segment
from .structure import HeadingHint, prepare_structure, resolve_contents


@dataclass
class Node:
    node_id: str
    kind: str
    title: str
    start: int
    end: int
    central: dict | None = None
    children: list["Node"] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def walk(self) -> Iterator["Node"]:
        yield self
        for child in self.children:
            yield from child.walk()

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        return cls(**{**data, "children": [cls.from_dict(child) for child in data["children"]]})


@dataclass
class DocumentIndex:
    source: str
    source_name: str
    root: Node
    metadata: dict
    decisions: list[dict]

    def __post_init__(self) -> None:
        self._nodes = {node.node_id: node for node in self.root.walk()}
        self._parents = {child.node_id: node.node_id for node in self.root.walk() for child in node.children}
        self._line_starts = [0] + [i + 1 for i, character in enumerate(self.source) if character == "\n"]

    def to_dict(self) -> dict:
        return {"schema_version": 2, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentIndex":
        if data.get("schema_version") not in (1, 2):
            raise ValueError("Unsupported index schema_version")
        source = data["source"]
        expected = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if data["metadata"]["source_sha256"] != expected:
            raise ValueError("Index source hash mismatch")
        root = Node.from_dict(data["root"])
        seen: set[str] = set()

        def validate(node: Node, start: int, end: int) -> None:
            if node.node_id in seen or not start <= node.start <= node.end <= end:
                raise ValueError("Invalid node ID or source span")
            seen.add(node.node_id)
            if node.central is not None:
                central = node.central
                if not node.start <= central["start"] < central["end"] <= node.end:
                    raise ValueError("Central sentence outside node")
                if central["text"] != source[central["start"]:central["end"]]:
                    raise ValueError("Central sentence does not match source")
            for child in node.children:
                validate(child, node.start, node.end)

        validate(root, 0, len(source))
        kinds = {node.node_id: node.kind for node in root.walk()}
        for node in root.walk():
            target = node.metadata.get("target_id")
            if target is not None and kinds.get(target) != "heading":
                raise ValueError("Contents target must reference a heading")
        return cls(source, data["source_name"], root, data["metadata"], data["decisions"])

    def _node(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise ValueError(f"Unknown node_id: {node_id}") from None

    def _summary(self, node: Node) -> dict:
        return {"node_id": node.node_id, "kind": node.kind, "title": node.title,
                "central_sentence": node.central["text"] if node.central else None,
                "child_count": len(node.children), "parent_id": self._parents.get(node.node_id),
                "metadata": node.metadata}

    def parent(self, node_id: str) -> dict | None:
        self._node(node_id)
        parent_id = self._parents.get(node_id)
        return self._summary(self._node(parent_id)) if parent_id else None

    def path(self, node_id: str) -> list[dict]:
        path = []
        node = self._node(node_id)
        while True:
            path.append({"node_id": node.node_id, "kind": node.kind, "title": node.title})
            parent_id = self._parents.get(node.node_id)
            if parent_id is None:
                break
            node = self._node(parent_id)
        return list(reversed(path))

    def outline(self) -> dict:
        return {"document": self._summary(self.root),
                "headings": [self._summary(node) for node in self.root.walk() if node.kind == "heading"],
                "contents": [self._summary(node) for node in self.root.walk() if node.kind == "toc_entry"]}

    def children(self, node_id: str = "root") -> list[dict]:
        """Give an LLM one level of navigable context, without full document text."""
        return [self._summary(child) for child in self._node(node_id).children]

    def read(self, node_id: str) -> dict:
        node = self._node(node_id)
        return {**self._summary(node), "text": self.source[node.start:node.end],
                "citation": {"source": self.source_name, "node_id": node.node_id,
                             "start": node.start, "end": node.end,
                             "start_line": bisect_right(self._line_starts, node.start),
                             "end_line": bisect_right(self._line_starts, max(node.start, node.end - 1)),
                             "source_sha256": self.metadata["source_sha256"]}}


def build_index(
    source: str, *, source_name: str = "document", scorer: Similarity | None = None,
    config: Config | None = None,
    heading_hints: list[HeadingHint] | None = None,
) -> DocumentIndex:
    scorer = scorer if scorer is not None else LexicalJaccard()
    config = config if config is not None else Config()
    root = Node("root", "document", source_name, 0, len(source))
    stack: list[tuple[int, Node]] = [(0, root)]
    pending: list[Block] = []
    decisions: list[dict] = []
    sequence = 0

    def create(kind: str, title: str, start: int, end: int, central=None, metadata=None) -> Node:
        nonlocal sequence
        sequence += 1
        return Node(f"n{sequence:06d}", kind, title, start, end, central, metadata=metadata or {})

    def flush() -> None:
        groups, trace = segment(source, pending, scorer, config)
        decisions.extend({"parent_id": stack[-1][1].node_id, **entry} for entry in trace)
        for group in groups:
            paragraph_spans = [sentence_spans(source, block) for block in group]
            representatives = central_sentences(
                source, paragraph_spans, scorer, config.sentence_budget, include_section=True,
            )
            central = representatives[0]
            section = create("section", central["text"] if central else "Section",
                             group[0].start, group[-1].end, central)
            stack[-1][1].children.append(section)
            for number, (block, spans) in enumerate(zip(group, paragraph_spans), start=1):
                paragraph_central = representatives[number]
                paragraph = create("paragraph", f"Paragraph {number}", block.start, block.end, paragraph_central)
                section.children.append(paragraph)
                for start, end in spans:
                    paragraph.children.append(create("sentence", source[start:end], start, end))
        pending.clear()

    for block in prepare_structure(source, heading_hints):
        if block.kind == "heading":
            flush()
            while stack[-1][0] >= block.level:
                _, closed = stack.pop()
                closed.end = block.start
            heading = create("heading", block.title, block.start, len(source), metadata=block.metadata)
            stack[-1][1].children.append(heading)
            stack.append((block.level, heading))
        elif block.kind == "toc":
            flush()
            contents = create("toc", "Contents entries", block.start, block.end,
                              metadata={"origin": "contents"})
            stack[-1][1].children.append(contents)
            for entry in block.metadata["entries"]:
                contents.children.append(create("toc_entry", entry["title"], entry["start"], entry["end"],
                                                metadata={"anchor": entry["anchor"],
                                                          "printed_page": entry["printed_page"]}))
        else:
            pending.append(block)
    flush()
    diagnostics = resolve_contents(root)
    first_heading = next((node for node in root.walk() if node.kind == "heading"
                          and node.metadata.get("role") != "contents"), None)
    if first_heading is not None and first_heading.start == len(source) - len(source.lstrip()):
        root.title = first_heading.title
        root.metadata["title_source_id"] = first_heading.node_id
    metadata = {"scorer": scorer.name, "config": asdict(config),
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "offset_unit": "unicode-code-point", "end_exclusive": True}
    metadata["structure"] = {"method": "declared-structure", "model_calls": 0,
                             "contents_diagnostics": diagnostics}
    scorer_metadata = getattr(scorer, "metadata", None)
    if callable(scorer_metadata):
        metadata["provider"] = scorer_metadata()
    return DocumentIndex(source, source_name, root, metadata, decisions)
