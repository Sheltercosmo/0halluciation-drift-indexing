import hashlib
import unittest
from zero_index.index import DocumentIndex,Node
from scripts.retrieval_v4_structure import restore_heading_hierarchy


def flat_index(titles):
    source=''.join(f'Paragraph {i}.\n' for i in range(len(titles)))
    root=Node('root','document','Paper',0,len(source));start=0
    for i,title in enumerate(titles):
        end=source.index('\n',start)
        central={'start':start,'end':end,'text':source[start:end]}
        p=Node(f'p{i}','paragraph','Paragraph',start,end,central)
        block=Node(f'b{i}','section','Block',start,end,central,children=[p])
        root.children.append(Node(f'h{i}','heading',title,start,end,children=[block]));start=end+1
    return DocumentIndex(source,'fixture',root,{'source_sha256':hashlib.sha256(source.encode()).hexdigest()},[])


class NativeHeadingHierarchy(unittest.TestCase):
    def test_nested_native_heading_is_not_a_root_competitor(self):
        old=flat_index(['Methods','Methods ::: Features','Results'])
        repaired=restore_heading_hierarchy(old)
        self.assertEqual(repaired.parent('h1')['node_id'],'h0')
        self.assertEqual([n.node_id for n in repaired.root.children],['h0','h2'])
        self.assertEqual([n['title'] for n in repaired.path('p1') if n['kind']=='heading'],['Methods','Features'])

    def test_missing_parent_is_structural_and_source_is_preserved(self):
        old=flat_index(['Methods ::: Features','Methods ::: Training ::: Loss','Results'])
        before=old.to_dict();repaired=restore_heading_hierarchy(old)
        self.assertEqual(len(repaired.root.children),2)
        self.assertEqual(repaired.parent('h0')['node_id'],repaired.root.children[0].node_id)
        self.assertEqual([n['title'] for n in repaired.path('p1') if n['kind']=='heading'],['Methods','Training','Loss'])
        for i in range(3):
            self.assertEqual(old._node(f'p{i}'),repaired._node(f'p{i}'))
            self.assertEqual(old.read(f'p{i}')['text'],repaired.read(f'p{i}')['text'])
        self.assertEqual(before,old.to_dict())
        DocumentIndex.from_dict(repaired.to_dict())
        self.assertEqual(repaired.to_dict(),restore_heading_hierarchy(repaired).to_dict())

    def test_repeated_noncontiguous_heading_does_not_merge_across_section(self):
        old=flat_index(['Methods ::: Features','Results','Methods ::: Features'])
        repaired=restore_heading_hierarchy(old)
        self.assertEqual(len(repaired.root.children),3)
        self.assertNotEqual(repaired.parent('h0')['node_id'],repaired.parent('h2')['node_id'])

    def test_flat_document_and_repeated_leaf_keep_distinct_nodes(self):
        old=flat_index(['Introduction','Methods','Methods'])
        repaired=restore_heading_hierarchy(old)
        self.assertEqual([n.node_id for n in repaired.root.children],['h0','h1','h2'])
        self.assertEqual(len([n for n in repaired.root.walk() if n.kind=='paragraph']),3)


if __name__=='__main__':unittest.main()
