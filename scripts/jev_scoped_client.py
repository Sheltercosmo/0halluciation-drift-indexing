"""Make per-candidate caching match the inputs of parallel Jev decisions.

Only the original question/search need is shared. Each decision receives its
own complete candidate in structured instructions, independent of batch peers
and position. The frozen v4 adapter remains unchanged for reproducibility.
"""
from scripts.retrieval_v4_clients import EvidenceJevV4


class ScopedEvidenceJev(EvidenceJevV4):
    def _request(self, state, questions):
        collections = [key for key in ('nodes', 'paragraphs') if key in state]
        if len(collections) != 1:
            raise ValueError('Expected one evidence candidate collection')
        collection = collections[0]
        prefix = 'n' if collection == 'nodes' else 'p'
        common = {key: value for key, value in state.items() if key != collection}
        scoped = {}
        for key, question in questions.items():
            if not key.startswith('q') or not key[1:].isdigit():
                raise ValueError('Unexpected evidence question identity')
            candidate = prefix + key[1:]
            path = collection + '.' + candidate
            instruction = question['instructions']
            if not isinstance(instruction, str) or path not in instruction:
                raise ValueError('Missing candidate reference')
            scoped[key] = {**question, 'instructions': {
                'target': state[collection][candidate],
                'decision': instruction.replace(path, '`target`'),
            }}
        return super()._request(common, scoped)
