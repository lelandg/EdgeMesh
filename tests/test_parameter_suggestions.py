import unittest

import numpy as np

from parameter_suggestions import suggest_parameters
from session_state import SessionDocument, SessionHistory, validate_settings


class SuggestionTests(unittest.TestCase):
    def test_local_deterministic_bounded_and_no_mutation(self):
        image = np.random.default_rng(4).integers(0, 256, (32, 24, 3), dtype=np.uint8)
        original = image.copy()
        settings = {"depth_amount": 2.0}
        suggestions = suggest_parameters(image, settings)
        self.assertEqual(suggestions, suggest_parameters(image, settings))
        self.assertEqual(settings, {"depth_amount": 2.0})
        np.testing.assert_array_equal(image, original)
        for suggestion in suggestions:
            validate_settings(dict(suggestion.settings))
            self.assertTrue(suggestion.reason)
            with self.assertRaises(TypeError):
                suggestion.settings["depth_amount"] = 90

    def test_accepted_suggestion_can_be_undone(self):
        history = SessionHistory()
        doc = SessionDocument("source.png", {"depth_amount": 2.0})
        history.record(doc)
        suggestion = next(x for x in suggest_parameters(np.zeros((3, 3, 3), np.uint8), doc.settings)
                          if "depth_amount" in x.settings)
        self.assertEqual(history.current.settings["depth_amount"], 2.0)
        doc.settings.update(suggestion.settings)
        history.record(doc)
        self.assertEqual(history.current.settings["depth_amount"], 1.0)
        self.assertEqual(history.undo().settings["depth_amount"], 2.0)

    def test_singleton_and_invalid_images(self):
        self.assertTrue(suggest_parameters(np.zeros((1, 1, 3), np.uint8), {}))
        for image in (np.zeros((2, 2)), np.zeros((0, 1, 3), np.uint8)):
            with self.assertRaises(ValueError):
                suggest_parameters(image, {})


if __name__ == "__main__":
    unittest.main()
