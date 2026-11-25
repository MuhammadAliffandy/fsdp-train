import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import diagnose_fsdp_system as diagnosis
import physical_fsdp as data
from test_fsdp_verified_cache import fixture


class SystemDiagnosisTests(unittest.TestCase):
    def test_roi_audit_roundtrip_physical_extent_and_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, prepared, _ = fixture(Path(directory))
            sample = prepared['samples']['train'][0]
            result = diagnosis.audit_sample(sample, prepared['policy']['normalization'])
            self.assertEqual(result['full_target_voxels'], 1)
            self.assertTrue(0 <= result['roundtrip_recomputed'] <= 1)
            self.assertEqual(result['roundtrip_recomputed'], sample['roundtrip_dice'])
            self.assertTrue(all(value > 0 for value in result['physical_target_extent_lps_mm']))
            self.assertTrue(0 <= result['detail_at_clip_upper_fraction'] <= 1)
            self.assertIn('sam_context_at_clip_upper_fraction', result)

    def test_inventory_keeps_click_protocols_distinct_and_handles_bad_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm, clicks in (('sam', [1, 5]), ('resunet', 0)):
                path = root / 'runs' / arm; path.mkdir(parents=True)
                data.save(path / 'summary.json', {'complete': True, 'experiment': arm, 'oracle_clicks': clicks,
                    'test_images_read': False, 'best_validation_metrics': {'mean_bilateral_patient_dice': .49}})
            broken = root / 'runs/broken'; broken.mkdir(); (broken / 'summary.json').write_text('{broken')
            result = diagnosis.experiment_inventory(root)
            self.assertEqual(len(result['runs']), 2)
            self.assertEqual(result['runs'][0]['oracle_clicks'], 0)
            self.assertEqual(len(result['unreadable_or_unsupported']), 1)

    def test_source_header_only_uses_selected_train_record(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, prepared, _ = fixture(Path(directory))
            record = prepared['source_provenance']['train'][27]
            result = diagnosis.source_header(record, source)
            self.assertEqual(result['case'], 'train_28')
            self.assertEqual(result['size'], [12] * 3)
            self.assertEqual(result['spacing_mm'], [.5] * 3)

    def test_candidate_is_separate_and_preserves_targets_grids_and_original_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, manifest, prepared, _ = fixture(root)
            before = prepared['samples']['train'][0]['image'].clone()
            policy = copy.deepcopy(prepared['policy'])
            output = root / 'candidate'; output.mkdir()
            report = diagnosis.build_normalization_candidate(prepared, source, output, 'original-cache-hash')
            candidate = torch.load(output / 'candidate_context_norm.pt', weights_only=False)
            self.assertFalse(report['training_launched'])
            self.assertIn('TRAIN', report['normalization']['source'])
            torch.testing.assert_close(prepared['samples']['train'][0]['image'], before, rtol=0, atol=0)
            self.assertEqual(prepared['policy'], policy)
            for split in ('train', 'val'):
                for old, new in zip(prepared['samples'][split], candidate['samples'][split]):
                    self.assertEqual(old['grid'], new['grid'])
                    self.assertTrue(np.array_equal(old['native_target'], new['native_target']))
                    torch.testing.assert_close(old['target'], new['target'], rtol=0, atol=0)
                    torch.testing.assert_close(old['sam_target'], new['sam_target'], rtol=0, atol=0)
            from fsdp_verified_cache import validate_targets
            self.assertTrue(validate_targets(candidate, manifest)['passed'])

    def test_probability_layer_probe_is_read_only_and_train_only(self):
        from test_physical_resunet import PhysicalResUNetTests
        from physical_resunet import PhysicalResUNet
        from evaluate_private_validation import file_hash
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root / 'run'; run.mkdir()
            prepared = PhysicalResUNetTests().fixture()
            config = {'cache_sha256': 'cache', 'code_sha256': {}}
            data.save(run / 'config.json', config)
            model = PhysicalResUNet((8, 8, 16, 16, 24))
            torch.save({'config': config, 'model': model.state_dict()}, run / 'resunet_best.pth')
            checksum = file_hash(run / 'resunet_best.pth')
            data.save(run / 'summary.json', {'complete': True, 'test_images_read': False, 'oracle_clicks': 0,
                'checkpoint_sha256': checksum, 'threshold_train_only': .95})
            with patch('physical_resunet.PhysicalResUNet', side_effect=lambda: PhysicalResUNet((8, 8, 16, 16, 24))), \
                 patch('train_physical_resunet.overlay'):
                result = diagnosis.resunet_probe(run, prepared, root / 'probe', 'cache')
            self.assertEqual(len(result['records']), 2)
            self.assertTrue(all(r['case'] == 'train_01' for r in result['records']))
            self.assertIn('encoder.0', result['records'][0]['layers'])
            self.assertEqual(result['optimizer_updates'], 0)
            self.assertEqual(file_hash(run / 'resunet_best.pth'), checksum)
            self.assertEqual(len(list((root / 'probe').glob('*probability.nii.gz'))), 2)


if __name__ == '__main__':
    unittest.main()
