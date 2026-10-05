import unittest
from scripts.live_tree_reader_v2 import read_case, validate_mcq


class MCQReaderTests(unittest.TestCase):
    def test_labelled_choices_and_answer_schema_without_gold(self):
        class Reader:
            def structured(self, instruction, payload, schema, validator, stage):
                self.payload, self.schema = payload, schema
                result = {'answer': 'C'}; validator(result); return result
        reader = Reader()
        self.assertEqual(read_case(reader, {'question': 'Which?', 'options': ['one', 'two', 'three', 'four']}, 'source'), 'C')
        self.assertEqual(reader.payload['options'], {'A': 'one', 'B': 'two', 'C': 'three', 'D': 'four'})
        self.assertEqual(reader.schema['properties']['answer']['enum'], ['A', 'B', 'C', 'D'])
        with self.assertRaises(ValueError): validate_mcq({'answer': 'Unanswerable'})


if __name__ == '__main__': unittest.main()
