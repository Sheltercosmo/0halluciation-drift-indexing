"""Small Markdown/plain-text parser preserving original character offsets."""

from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class Block:
    kind: str
    start: int
    end: int
    title: str = ""
    level: int = 0
    verbatim: bool = False
    metadata: dict = field(default_factory=dict)


def parse_blocks(source: str) -> list[Block]:
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    blocks: list[Block] = []
    pending: int | None = None

    def flush(end: int) -> None:
        nonlocal pending
        if pending is not None:
            start = offsets[pending]
            while start < end and source[start].isspace():
                start += 1
            while end > start and source[end - 1].isspace():
                end -= 1
            if start < end:
                blocks.append(Block("paragraph", start, end))
            pending = None

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\r\n")
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            flush(offsets[i])
            marker = fence.group(1)
            start = i
            i += 1
            closing = r"^ {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*$"
            while i < len(lines):
                if re.match(closing, lines[i].rstrip("\r\n")):
                    i += 1
                    break
                i += 1
            end = offsets[i]
            while end > offsets[start] and source[end - 1] in "\r\n":
                end -= 1
            blocks.append(Block("paragraph", offsets[start], end, verbatim=True))
            continue
        heading = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)|[ \t]*)$", line)
        setext = (
            pending is None and line.strip() and i + 1 < len(lines)
            and re.fullmatch(r" {0,3}(?:=+|-+)\s*", lines[i + 1].rstrip("\r\n"))
        )
        if heading:
            flush(offsets[i])
            title = re.sub(r"\s+#+\s*$", "", heading.group(2) or "").strip()
            blocks.append(Block("heading", offsets[i], offsets[i + 1], title, len(heading.group(1))))
        elif setext:
            level = 1 if lines[i + 1].lstrip().startswith("=") else 2
            blocks.append(Block("heading", offsets[i], offsets[i + 2], line.strip(), level))
            i += 1
        elif not line.strip():
            flush(offsets[i])
        elif pending is None:
            pending = i
        i += 1
    flush(len(source))
    return blocks


def sentence_spans(source: str, block: Block) -> list[tuple[int, int]]:
    if block.verbatim:
        return [(block.start, block.end)]
    text = source[block.start:block.end]
    spans: list[tuple[int, int]] = []
    start = 0
    # Deliberately simple sentence heuristic; abbreviations need a richer parser.
    for boundary in re.finditer(r"[.!?。！？]+(?:[\"'”’)]*)(?:\s+|$)", text):
        end = boundary.end()
        fragment = text[start:end]
        left = start + len(fragment) - len(fragment.lstrip())
        right = start + len(fragment.rstrip())
        if left < right:
            spans.append((block.start + left, block.start + right))
        start = end
    if start < len(text):
        fragment = text[start:]
        left = start + len(fragment) - len(fragment.lstrip())
        right = start + len(fragment.rstrip())
        if left < right:
            spans.append((block.start + left, block.start + right))
    return spans
