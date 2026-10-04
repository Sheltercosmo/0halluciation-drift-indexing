import unittest

from scripts.frontier_data import validate_coverage


class CoverageTests(unittest.TestCase):
    def test_failures_stay_in_denominator(self):
        expected = [{"id": "easy"}, {"id": "hard"}]
        predictions = [{"id": "easy", "status": "ok"}, {"id": "hard", "status": "failed"}]
        self.assertEqual(validate_coverage(expected, predictions), {"expected": 2, "recorded": 2, "failed": 1})

    def test_omitted_hard_case_is_not_a_complete_run(self):
        with self.assertRaisesRegex(ValueError, "missing=1"):
            validate_coverage([{"id": "a"}, {"id": "b"}], [{"id": "a", "status": "ok"}])

    def test_duplicate_cannot_replace_missing_case(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_coverage([{"id": "a"}, {"id": "b"}],
                              [{"id": "a", "status": "ok"}, {"id": "a", "status": "ok"}])

    def test_unexpected_ids_and_missing_status_rejected(self):
        for predictions in ([{"id": "z", "status": "ok"}], [{"id": "a"}]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                validate_coverage([{"id": "a"}], predictions)


if __name__ == "__main__":
    unittest.main()
