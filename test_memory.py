import unittest

import numpy as np

from memory import MergeMemory, RecentMemory, pool_patches


PATCH = np.zeros((4, 4, 3), dtype=np.uint8)


class VisualMemoryTests(unittest.TestCase):
    def test_pooling_keeps_image_regions_in_place(self):
        patches = np.zeros((16, 16, 1), dtype=np.float32)
        patches[:4, :4] = 1
        patches[12:, 12:] = 2

        pooled = pool_patches(patches)

        self.assertEqual(pooled.shape, (4, 4, 1))
        self.assertEqual(float(pooled[0, 0, 0]), 1)
        self.assertEqual(float(pooled[3, 3, 0]), 2)
        self.assertEqual(float(pooled[0, 3, 0]), 0)

    def test_novel_patch_gets_a_slot_by_merging_similar_old_patches(self):
        memory = MergeMemory(capacity=2)
        memory.observe([1, 0], PATCH, frame=0)
        memory.observe([0.98, 0.2], PATCH, frame=1)
        memory.observe([0, 1], PATCH, frame=2)

        self.assertEqual(len(memory.entries), 2)
        self.assertEqual(sorted(entry.mass for entry in memory.entries), [1, 2])
        match = memory.search([0, 1], before_frame=3)
        self.assertEqual(match.entry.first_frame, 2)
        self.assertAlmostEqual(match.score, 1.0)

    def test_merged_memory_remembers_an_older_family_after_recent_only_forgets_it(self):
        merged = MergeMemory(capacity=2)
        recent = RecentMemory(capacity=2)
        for frame, vector in enumerate(([1, 0, 0], [0, 1, 0], [0, 0.99, 0.14], [0, 0.98, 0.2])):
            merged.observe(vector, PATCH, frame)
            recent.observe(vector, PATCH, frame)

        self.assertAlmostEqual(merged.search([1, 0, 0], before_frame=4).score, 1.0)
        self.assertAlmostEqual(recent.search([1, 0, 0], before_frame=4).score, 0.0)

    def test_memory_stays_bounded_and_ignores_current_frame_when_searched(self):
        memory = MergeMemory(capacity=3)
        for frame in range(20):
            memory.observe([np.cos(frame), np.sin(frame)], PATCH, frame)

        self.assertEqual(len(memory.entries), 3)
        self.assertIsNone(memory.search([1, 0], before_frame=0))

    def test_snapshot_queries_do_not_see_the_frame_being_shown(self):
        memory = MergeMemory(capacity=1)
        memory.observe([1, 0], PATCH, frame=0)
        past = memory.snapshot()
        memory.observe([0, 1], PATCH, frame=1)

        self.assertAlmostEqual(memory.search([0, 1], entries=past).score, 0.0)
        self.assertGreater(memory.search([0, 1]).score, 0.6)


if __name__ == "__main__":
    unittest.main()
