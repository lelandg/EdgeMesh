"""History navigation and the user's preview-before-restore workflow."""

import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication

from history_panel import HistoryPanel
from session_state import SessionDocument, SessionHistory


def document(resolution, **kwargs):
    return SessionDocument(
        "C:/private/project/photo.png",
        {"resolution": resolution, "model": "DPT", "depth_amount": 0.8},
        **kwargs,
    )


class HistoryNavigationTests(unittest.TestCase):
    def test_select_is_bounded_and_record_replaces_future_states(self):
        history = SessionHistory(limit=3)
        for resolution in (100, 200, 300, 400):
            history.record(document(resolution))
        self.assertEqual(history.index, 2)
        self.assertEqual([row["settings"]["resolution"] for row in history.summaries()], [200, 300, 400])
        candidate = history.snapshot_at(0)
        self.assertEqual(candidate.settings["resolution"], 200)
        self.assertEqual(history.index, 2, "Previewing a candidate must not commit restoration")
        history.select(0)
        history.record(document(250))
        self.assertEqual([row["settings"]["resolution"] for row in history.summaries()], [200, 250])
        self.assertFalse(history.can_redo)
        for invalid in (-1, 2, True, 0.0, None):
            for operation in (history.snapshot_at, history.select):
                with self.subTest(invalid=invalid, operation=operation), self.assertRaises(IndexError):
                    operation(invalid)
                self.assertEqual(history.index, 1)

    def test_returned_snapshots_and_summaries_are_independent(self):
        history = SessionHistory()
        history.record(document(100, model_info={"nested": {"version": "original"}},
                                mask=np.ones((2, 3), dtype=bool), history=[{"action": "original"}]))
        candidate = history.snapshot_at(0)
        candidate.settings["resolution"] = 200
        candidate.model_info["nested"]["version"] = "changed"
        candidate.mask[0, 0] = False
        candidate.history[0]["action"] = "changed"
        summary = history.summaries()[0]
        summary["settings"]["resolution"] = 300
        summary["model_info"]["nested"]["version"] = "also changed"
        actual = history.current
        self.assertEqual(actual.settings["resolution"], 100)
        self.assertEqual(actual.model_info["nested"]["version"], "original")
        self.assertTrue(actual.mask.all())
        self.assertEqual(actual.history, [{"action": "original"}])

    def test_summary_reports_changes_without_decoding_masks(self):
        history = SessionHistory()
        history.record(document(100, mask=np.ones((2, 3), dtype=bool)))
        history.record(document(200))
        with patch("session_state._decode_mask", side_effect=AssertionError("Unexpected mask decode")):
            summaries = history.summaries()
        self.assertEqual(summaries[0]["mask_shape"], (2, 3))
        self.assertTrue(summaries[1]["mask_changed"])
        self.assertFalse(summaries[1]["has_mask"])
        self.assertEqual(summaries[1]["changed_settings"], {"resolution": {"before": 100, "after": 200}})

    def test_empty_history_rejects_selection(self):
        history = SessionHistory()
        self.assertEqual(history.index, -1)
        self.assertEqual(history.summaries(), [])
        with self.assertRaises(IndexError):
            history.select(0)
        with self.assertRaises(IndexError):
            history.snapshot_at(0)


class HistoryPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.history = SessionHistory()
        self.history.record(document(100, mask=np.ones((2, 3), dtype=bool)))
        self.history.record(document(200))
        self.panel = HistoryPanel()

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()

    def test_refresh_selection_and_restore_signal_do_not_move_cursor(self):
        restored = []
        self.panel.restore_requested.connect(restored.append)
        with patch("session_state._decode_mask", side_effect=AssertionError("Unexpected mask decode")):
            self.panel.refresh(self.history)
        self.assertEqual(self.panel.list.topLevelItemCount(), 2)
        self.assertIn("Current", self.panel.list.topLevelItem(1).text(0))
        self.assertFalse(self.panel.restore_button.isEnabled())
        self.panel.list.setCurrentItem(self.panel.list.topLevelItem(0))
        self.assertTrue(self.panel.restore_button.isEnabled())
        self.assertEqual(restored, [], "Inspecting a state must not restore it")
        self.panel.restore_button.click()
        self.assertEqual(restored, [0])
        self.assertEqual(self.history.index, 1)
        self.history.select(0)
        self.panel.refresh(self.history)
        self.assertIn("Current", self.panel.list.topLevelItem(0).text(0))
        self.assertFalse(self.panel.restore_button.isEnabled())

    def test_details_show_changes_and_source_names_without_private_paths(self):
        self.panel.refresh(self.history)
        details = self.panel.details.toPlainText()
        self.assertIn("Resolution: 100 → 200", details)
        self.assertIn("Mask removed", details)
        self.assertIn("photo.png", details)
        self.assertNotIn("private", details)
        self.assertEqual(self.panel.list.topLevelItem(0).text(1), "photo.png")
        self.panel.list.setCurrentItem(self.panel.list.topLevelItem(0))
        self.assertIn("3 × 2 pixels", self.panel.details.toPlainText())

    def test_eviction_selects_current_state_but_redundant_refresh_preserves_inspection(self):
        history = SessionHistory(limit=3)
        for resolution in (100, 200, 300):
            history.record(document(resolution))
        self.panel.refresh(history)
        self.panel.list.setCurrentItem(self.panel.list.topLevelItem(0))
        self.panel.refresh(history)
        self.assertEqual(self.panel.list.currentItem().text(3), "100")

        # At capacity, a new state keeps the cursor at 2 while evicting row 0.
        history.record(document(400))
        self.assertEqual(history.index, 2)
        self.panel.refresh(history)
        self.assertEqual(self.panel.list.topLevelItem(0).text(3), "200")
        self.assertEqual(self.panel.list.currentItem().text(3), "400")
        self.assertIn("Current", self.panel.list.currentItem().text(0))
        self.assertFalse(self.panel.restore_button.isEnabled())

    def test_empty_history_clears_stale_selection_and_details(self):
        self.panel.refresh(self.history)
        self.panel.refresh(SessionHistory())
        self.assertEqual(self.panel.list.topLevelItemCount(), 0)
        self.assertEqual(self.panel.details.toPlainText(), "")
        self.assertFalse(self.panel.restore_button.isEnabled())
        self.assertIn("No saved states", self.panel.status_label.text())


if __name__ == "__main__":
    unittest.main()
