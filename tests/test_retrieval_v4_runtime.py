import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.run_retrieval_v4 import retry_atomic_replaces


class AtomicReplacementRetries(unittest.TestCase):
    def test_only_the_same_file_operation_is_retried(self):
        error=PermissionError('sharing conflict');error.winerror=32
        with patch.object(Path,'replace',side_effect=[error,error,Path('destination')]) as replacement:
            with patch('scripts.run_retrieval_v4.time.sleep'):
                with retry_atomic_replaces():self.assertEqual(Path('temporary').replace('destination'),Path('destination'))
            self.assertEqual(replacement.call_count,3)
            self.assertTrue(all(c.args==(Path('temporary'),'destination') for c in replacement.call_args_list))

    def test_unrelated_errors_are_not_retried(self):
        with patch.object(Path,'replace',side_effect=PermissionError('permission denied')) as replacement:
            with self.assertRaises(PermissionError):
                with retry_atomic_replaces():Path('temporary').replace('destination')
            self.assertEqual(replacement.call_count,1)


if __name__=='__main__':unittest.main()
