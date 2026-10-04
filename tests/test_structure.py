import unittest

from zero_index import DocumentIndex, HeadingHint, build_index
from zero_index.structure import prepare_structure


class NoModel:
    name = "no-model"

    def score(self, *args):
        raise AssertionError("Structural text must never reach Jev")

    def representatives(self, *args):
        raise AssertionError("Structural text must never reach Jev")


class StructureTests(unittest.TestCase):
    def test_contents_and_titles_use_no_model_and_link_to_existing_headings(self):
        source = "# Manual\n\n## Contents\n\n- [Setup](#setup)\n- [Usage](#usage)\n\n## Setup\n\n## Usage\n"
        index = build_index(source, scorer=NoModel())
        self.assertEqual(index.root.title, "Manual")
        outline = index.outline()
        self.assertEqual(len(outline["contents"]), 2)
        self.assertEqual(len(outline["headings"]), 4)
        for entry in outline["contents"]:
            self.assertEqual(entry["metadata"]["resolution"], "resolved")
            target = index._node(entry["metadata"]["target_id"])
            self.assertEqual(target.title, entry["title"])
        self.assertFalse(any(node.kind == "section" for node in index.root.walk()))
        self.assertEqual(index.metadata["structure"]["model_calls"], 0)
        self.assertEqual(DocumentIndex.from_dict(index.to_dict()).to_dict(), index.to_dict())

    def test_duplicate_titles_disambiguate_explicit_anchors_but_not_printed_labels(self):
        text = "# Contents\n\n- [Second](#usage-1)\nUsage .... 4\nMissing .... 9\n\n# Usage\n\n# Usage\n"
        index = build_index(text, scorer=NoModel())
        entries = index.outline()["contents"]
        self.assertEqual([e["metadata"]["resolution"] for e in entries], ["resolved", "ambiguous", "unresolved"])
        self.assertEqual(index._node(entries[0]["metadata"]["target_id"]).metadata["anchor"], "usage-1")
        self.assertEqual(entries[1]["metadata"]["printed_page"], "4")

    def test_literal_slug_collision_stays_unique(self):
        index = build_index("# Usage\n\n# Usage-1\n\n# Usage", scorer=NoModel())
        self.assertEqual([h["metadata"]["anchor"] for h in index.outline()["headings"]], ["usage", "usage-1", "usage-2"])

    def test_mixed_prose_and_code_are_not_toc_entries(self):
        text = "# Contents\n\nThis section discusses contents.\n\n```\n- [Setup](#setup)\n```\n\n# Setup"
        blocks = prepare_structure(text)
        self.assertFalse(any(block.kind == "toc" for block in blocks))
        self.assertTrue(any(block.verbatim for block in blocks))

    def test_heading_hints_split_text_and_preserve_provenance_and_hierarchy(self):
        text = "Manual\n\n1 Setup\nStart here.\n\n1.1 Options\nChoose carefully."
        hints = [HeadingHint(0, 6, "Manual", 1, "docling", "#/texts/0", 1),
                 HeadingHint(text.index("1 Setup"), text.index("Start"), "1 Setup", 2, "pdf-outline"),
                 HeadingHint(text.index("1.1 Options"), text.index("Choose"), "1.1 Options", 3, "docling")]
        index = build_index(text, heading_hints=hints, scorer=NoModel())
        headings = index.outline()["headings"]
        self.assertEqual([h["metadata"]["heading_level"] for h in headings], [1, 2, 3])
        self.assertEqual(headings[0]["metadata"]["source_ref"], "#/texts/0")
        self.assertEqual(index.parent(headings[-1]["node_id"])["node_id"], headings[-2]["node_id"])
        leaves = [n for n in index.root.walk() if n.kind == "sentence"]
        self.assertEqual([index.read(n.node_id)["text"] for n in leaves], ["Start here.", "Choose carefully."])

    def test_native_markdown_levels_win_over_imported_hints(self):
        index = build_index("## Heading\n", heading_hints=[HeadingHint(0, 11, "Heading", 5)])
        self.assertEqual(index.outline()["headings"][0]["metadata"]["heading_level"], 2)

    def test_bad_hints_fail_instead_of_inventing_headings(self):
        for hint in (HeadingHint(0, 5, "Wrong", 1), HeadingHint(1, 5, "itle", 1),
                     HeadingHint(0, 5, "Title", 0), HeadingHint(0, 500, "Title", 1)):
            with self.subTest(hint=hint), self.assertRaises(ValueError):
                build_index("Title\n\nBody.", heading_hints=[hint])

    def test_parent_chain_and_source_containment(self):
        index = build_index("# A\n\n## B\n\nSpecific fact.")
        leaf = next(n for n in index.root.walk() if n.kind == "sentence")
        self.assertEqual([p["kind"] for p in index.path(leaf.node_id)],
                         ["document", "heading", "heading", "section", "paragraph", "sentence"])
        self.assertIsNone(index.parent("root"))

    def test_legacy_schema_can_still_be_loaded(self):
        data = build_index("Old text.").to_dict()
        data["schema_version"] = 1
        def strip(node):
            node.pop("metadata", None)
            for child in node["children"]:
                strip(child)
        strip(data["root"])
        self.assertEqual(DocumentIndex.from_dict(data).read("root")["text"], "Old text.")

    def test_invalid_contents_reference_is_rejected_on_load(self):
        data = build_index("# Contents\n\n- [Target](#target)\n\n# Target").to_dict()
        entry = data["root"]["children"][0]["children"][0]["children"][0]
        entry["metadata"]["target_id"] = "absent"
        with self.assertRaises(ValueError):
            DocumentIndex.from_dict(data)


if __name__ == "__main__":
    unittest.main()
