"""Unit tests for BPM sanitization in simple_ui."""

import unittest

from acestep.simple_ui import _sanitize_bpm_for_ui


class SimpleUiBpmSanitizationTests(unittest.TestCase):
    """Validate BPM normalization used by simple UI fields."""

    def test_returns_none_for_missing_or_non_numeric_values(self):
        """Missing and non-numeric values should map to None."""
        self.assertIsNone(_sanitize_bpm_for_ui(None))
        self.assertIsNone(_sanitize_bpm_for_ui(""))
        self.assertIsNone(_sanitize_bpm_for_ui("N/A"))
        self.assertIsNone(_sanitize_bpm_for_ui("auto"))
        self.assertIsNone(_sanitize_bpm_for_ui("abc"))

    def test_returns_none_for_out_of_bounds_values(self):
        """Out-of-range BPM values should map to None for UI safety."""
        self.assertIsNone(_sanitize_bpm_for_ui(0))
        self.assertIsNone(_sanitize_bpm_for_ui(29))
        self.assertIsNone(_sanitize_bpm_for_ui(301))

    def test_accepts_in_range_values(self):
        """Valid BPM values should be returned as ints."""
        self.assertEqual(_sanitize_bpm_for_ui(30), 30)
        self.assertEqual(_sanitize_bpm_for_ui(120), 120)
        self.assertEqual(_sanitize_bpm_for_ui("128"), 128)
        self.assertEqual(_sanitize_bpm_for_ui(145.8), 145)


if __name__ == "__main__":
    unittest.main()
