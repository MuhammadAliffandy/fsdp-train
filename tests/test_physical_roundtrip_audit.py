import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import physical_fsdp as data
from audit_physical_roundtrip import inspect
from test_physical_fsdp import sample_fixture


class RoundtripAuditTests(unittest.TestCase):
    def test_zero_phase_reproduces_original_and_sources_unchanged(self):
        image, ct, mask, norm = sample_fixture()
        sample = data.make_sample(image, ct, mask, 'train_26', 'RFSDP',
            {'shape': [48, 48, 64], 'spacing': [.25] * 3}, norm)
        before = sample['target'].clone()
        labels = mask.astype(np.uint8)
        rows = inspect(image, labels, 1, sample)
        self.assertEqual(len(rows), 9)
        self.assertAlmostEqual(rows[0]['linear_signed_label_roundtrip']['dice'], sample['roundtrip_dice'])
        self.assertTrue((sample['target'] == before).all())
        np.testing.assert_array_equal(data.xyz(image), ct)
        np.testing.assert_array_equal(labels, mask)
        self.assertTrue(all(0 <= r['nearest_label_roundtrip']['dice'] <= 1 for r in rows))


if __name__ == '__main__':
    unittest.main()
