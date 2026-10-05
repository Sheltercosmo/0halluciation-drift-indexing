"""Isolated structure repair for future experiments; never used by frozen v3."""
from zero_index.index import DocumentIndex,Node


def restore_heading_hierarchy(index):
    """Rebuild contiguous native ``A ::: B`` paths without touching source leaves.

    An absent ancestor becomes a structural node with no generated summary.
    Repeated paths after a different section are separate occurrences. Existing
    heading IDs, topic groups, paragraph IDs, central sentences and source spans
    stay intact except that ancestor heading spans expand to contain children.
    """
    result=DocumentIndex.from_dict(index.to_dict())
    if result.metadata.get('native_hierarchy_repair')=='v4':return result
    if any(child.kind=='heading' for heading in result.root.children for child in heading.children):
        raise ValueError('Expected flat native-heading adapter input')
    old_children=result.root.children;result.root.children=[]
    used={node.node_id for node in index.root.walk()};sequence=0;active=[]
    for heading in old_children:
        if heading.kind!='heading':
            result.root.children.append(heading);active=[];continue
        native_title=heading.title
        parts=[part.strip() for part in native_title.split(' ::: ')]
        if any(not part for part in parts):parts=[native_title]
        common=0
        # The final component is a new native section, even if its title repeats.
        while common<min(len(active),len(parts)-1) and active[common][0]==parts[common]:common+=1
        active=active[:common]
        for depth in range(common,len(parts)):
            parent=active[-1][1] if active else result.root
            if depth==len(parts)-1:
                node=heading;node.title=parts[depth]
                node.metadata.update(native_heading_title=native_title,native_heading_path=parts)
            else:
                while True:
                    sequence+=1;node_id=f'v4h{sequence}'
                    if node_id not in used:break
                used.add(node_id)
                node=Node(node_id,'heading',parts[depth],heading.start,heading.end,
                          metadata={'origin':'native_heading_path','implicit_ancestor':True,
                                    'native_heading_path':parts[:depth+1]})
            parent.children.append(node);active.append((parts[depth],node))
    def expand(node):
        for child in node.children:expand(child)
        if node.kind=='heading' and node.children:
            node.start=min(node.start,*(c.start for c in node.children))
            node.end=max(node.end,*(c.end for c in node.children))
    expand(result.root)
    result.metadata.update(native_hierarchy_repair='v4',
        structure='dataset-native nested heading paths and unchanged source paragraph offsets')
    return DocumentIndex.from_dict(result.to_dict())
