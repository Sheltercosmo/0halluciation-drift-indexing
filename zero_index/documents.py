"""Source-preserving documents and native heading/paragraph trees."""
import hashlib
from pathlib import Path
from .index import DocumentIndex, Node
from .parse import Block, parse_blocks, sentence_spans


def document_from_text(text, *, source_name="document.md", title=None):
    """Keep Markdown/plain-text offsets, headings and fenced paragraphs intact."""
    if not isinstance(text, str):
        raise ValueError("Source text must be a string")
    units, headings = [], []
    section = 0
    first_title = None
    for block in parse_blocks(text):
        if block.kind == "heading":
            first_title = first_title or block.title
            while headings and headings[-1][0] >= block.level:
                headings.pop()
            headings.append((block.level, block.title))
            section += 1
        elif block.kind == "paragraph":
            units.append({"paragraph": len(units), "section": section,
                          "heading": " ::: ".join(name for _, name in headings),
                          "start": block.start, "end": block.end, "verbatim": block.verbatim})
    return {"id": source_name, "title": title or first_title or source_name, "text": text, "units": units}


def load_document(path):
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as stream:
        return document_from_text(stream.read(), source_name=path.name)


def build_document(doc_id, title, sections):
    text, units = "", []
    for section_id, (heading, paragraphs) in enumerate(sections):
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            start = len(text)
            text += paragraph
            units.append({"start": start, "end": len(text), "heading": heading or "",
                          "section": section_id, "paragraph": len(units)})
            text += "\n\n"
    return {"id": doc_id, "title": title, "text": text, "units": units}


def sections(doc):
    groups = []
    for unit in doc["units"]:
        if not groups or groups[-1][0]["section"] != unit["section"]:
            groups.append([])
        groups[-1].append(unit)
    return groups

def tree(doc, groups, partition):
    source = doc['text']
    root = Node('root', 'document', doc['title'], 0, len(source))
    headings = {}
    for number, group in enumerate(groups):
        native = group[0]['section']
        if native not in headings:
            units = [u for u in doc['units'] if u['section'] == native]
            headings[native] = Node(f'h{native}', 'heading', group[0]['heading'] or doc['title'],
                                     units[0]['start'], units[-1]['end'])
            root.children.append(headings[native])
        section = Node(f'b{number}', 'section', 'Section', group[0]['start'], group[-1]['end'])
        headings[native].children.append(section)
        for unit in group:
            paragraph = Node(f'p{unit["paragraph"]}', 'paragraph', f'Paragraph {unit["paragraph"] + 1}',
                             unit['start'], unit['end'])
            section.children.append(paragraph)
            for i, (a, b) in enumerate(sentence_spans(source, Block('paragraph', unit['start'], unit['end'], verbatim=unit.get('verbatim', False)))):
                paragraph.children.append(Node(f'{paragraph.node_id}s{i}', 'sentence', source[a:b], a, b))
    result = DocumentIndex(source, doc['id'], root, {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                           'partition': partition, 'config': {}, 'structure': 'dataset-native headings and paragraph offsets'}, [])
    return DocumentIndex.from_dict(result.to_dict())
