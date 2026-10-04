"""Declared document structure: source headings, parser hints, and TOC links.

No scorer or model dependency belongs in this module.
"""

from dataclasses import dataclass, replace
import re
import unicodedata
from urllib.parse import unquote

from .parse import Block, parse_blocks


@dataclass(frozen=True)
class HeadingHint:
    """A caller-validated heading in the SAME decoded source coordinate system."""

    start: int
    end: int
    title: str
    level: int
    origin: str = "parser"
    source_ref: str | None = None
    page: int | None = None


def normalized_title(title: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", title).casefold().split())


def heading_slug(title: str) -> str:
    # A documented local convention, not a promise of every Markdown renderer's rules.
    title = re.sub(r"[^\w\s-]", "", unicodedata.normalize("NFKC", title).casefold())
    return re.sub(r"\s", "-", title.strip())


def _entries(source: str, block: Block) -> list[dict]:
    entries = []
    cursor = block.start
    for line in source[block.start:block.end].splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            cursor += len(line)
            continue
        link = re.fullmatch(r"(?:[-*+]\s+|\d+[.)]\s+)?\[([^\]]+)\]\(#([^\s)]+)\)", stripped)
        printed = re.fullmatch(r"(.+?)\s*\.{2,}\s*([0-9]+|[ivxlcdmIVXLCDM]+)", stripped)
        if not link and not printed:
            return []  # A mixed prose block stays ordinary content.
        start = cursor + len(line) - len(line.lstrip())
        end = cursor + len(line.rstrip())
        entries.append({"start": start, "end": end,
                        "title": (link or printed).group(1).strip(),
                        "anchor": unquote(link.group(2)) if link else None,
                        "printed_page": printed.group(2) if printed else None})
        cursor += len(line)
    return entries


def prepare_structure(source: str, hints: list[HeadingHint] | None = None) -> list[Block]:
    blocks = parse_blocks(source)
    hints = sorted(hints or [], key=lambda hint: hint.start)
    previous_end = -1
    for hint in hints:
        if (type(hint.start) is not int or type(hint.end) is not int
                or not 0 <= hint.start < hint.end <= len(source)
                or type(hint.level) is not int or not 1 <= hint.level <= 64):
            raise ValueError("Invalid heading hint span or level")
        if hint.start < previous_end:
            raise ValueError("Heading hints cannot overlap")
        previous_end = hint.end
        if not hint.title.strip() or not hint.origin.strip():
            raise ValueError("Heading hints need a title and origin")
        if hint.start and source[hint.start - 1] != "\n":
            raise ValueError("Heading hints must begin at a line boundary")
        if hint.end < len(source) and source[hint.end - 1] != "\n" and source[hint.end] not in "\r\n":
            raise ValueError("Heading hints must end at a line boundary")
        overlaps = [block for block in blocks if block.start < hint.end and hint.start < block.end]
        if len(overlaps) != 1:
            raise ValueError("Heading hint must address exactly one parsed block")
        block = overlaps[0]
        if block.kind == "heading":
            if hint.start != block.start or normalized_title(hint.title) != normalized_title(block.title):
                raise ValueError("Heading hint conflicts with an explicit source heading")
            # Native heading markup wins over imported hierarchy hints.
            continue
        if block.verbatim or source[hint.start:hint.end].strip() != hint.title:
            raise ValueError("Heading hint must match source text outside fenced code")
        replacements = []
        for start, end in ((block.start, hint.start), (hint.end, block.end)):
            fragments = [replace(item, start=item.start + start, end=item.end + start)
                         for item in parse_blocks(source[start:end])] if start < end else []
            if start == block.start:
                replacements.extend(fragments)
                replacements.append(Block("heading", hint.start, hint.end, hint.title, hint.level,
                                          metadata={"origin": hint.origin, "source_ref": hint.source_ref,
                                                    "page": hint.page}))
            else:
                replacements.extend(fragments)
        position = blocks.index(block)
        blocks[position:position + 1] = replacements

    result = []
    toc_heading: int | None = None
    for block in blocks:
        if block.kind == "heading":
            block = replace(block, metadata={"origin": "markdown", **block.metadata,
                                            "heading_level": block.level,
                                            "heading_start": block.start, "heading_end": block.end})
            toc_heading = len(result) if normalized_title(block.title) in ("contents", "table of contents", "目录") else None
        elif toc_heading is not None and not block.verbatim:
            entries = _entries(source, block)
            if entries:
                block = replace(block, kind="toc", metadata={"origin": "contents", "entries": entries})
                heading = result[toc_heading]
                result[toc_heading] = replace(heading, metadata={**heading.metadata, "role": "contents"})
        result.append(block)
    return result


def resolve_contents(root) -> list[dict]:
    """Resolve navigation references only; never manufacture target headings."""
    headings = [node for node in root.walk() if node.kind == "heading"]
    by_slug, by_title, used = {}, {}, set()
    for heading in headings:
        base = heading_slug(heading.title)
        slug, suffix = base, 0
        while slug in used:
            suffix += 1
            slug = f"{base}-{suffix}"
        used.add(slug)
        by_slug[slug] = heading
        by_title.setdefault(normalized_title(heading.title), []).append(heading)
        heading.metadata["anchor"] = slug
    diagnostics = []
    for entry in (node for node in root.walk() if node.kind == "toc_entry"):
        anchor = entry.metadata.get("anchor")
        matches = ([by_slug[anchor]] if anchor in by_slug else []) if anchor is not None else by_title.get(normalized_title(entry.title), [])
        if len(matches) == 1 and matches[0].metadata.get("role") != "contents":
            entry.metadata.update(target_id=matches[0].node_id, resolution="resolved")
        else:
            status = "ambiguous" if len(matches) > 1 else "unresolved"
            entry.metadata.update(target_id=None, resolution=status)
            diagnostics.append({"node_id": entry.node_id, "title": entry.title, "status": status})
    return diagnostics
