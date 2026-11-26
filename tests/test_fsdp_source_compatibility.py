"""Accept only audited trainer revisions; model and metric changes remain fatal."""
import ast
import hashlib
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch
scripts = Path(__file__).resolve().parents[1] / 'scripts'
base_tree = ast.parse((scripts / 'train_sam_refine32.py').read_text())
code_files = next(ast.literal_eval(n.value) for n in base_tree.body
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CODE_FILES' for t in n.targets))
source = scripts / 'train_fsdp_research.py'
tree = ast.parse(source.read_text())
names = {'LEGACY_BASE_TRAINERS', 'COMPATIBLE_BASE_TRAINER', 'inspect_source_code'}
nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
         or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
namespace = dict(__file__=str(source), Path=Path, base=SimpleNamespace(CODE_FILES=code_files),
                 file_hash=lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
trainer = SimpleNamespace(**namespace)
trainer.code_signature = lambda: {n: namespace['file_hash'](scripts / n)
    for n in (*code_files, 'fsdp_research_model.py', 'fsdp_research_metrics.py')}


class SourceCompatibilityTests(unittest.TestCase):
    def test_legacy_trainer_is_recorded_and_requires_reproduction(self):
        hashes = trainer.code_signature()
        for old in trainer.LEGACY_BASE_TRAINERS:
            hashes['train_sam_refine32.py'] = old
            result = trainer.inspect_source_code(hashes)
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0]['require_source_metric_reproduction'])

    def test_unknown_trainer_model_or_metric_hash_is_rejected_with_filename(self):
        for name in ('train_sam_refine32.py', 'sam_refine32.py', 'fsdp_research_metrics.py'):
            hashes = trainer.code_signature()
            hashes[name] = 'unknown'
            with self.assertRaisesRegex(ValueError, name):
                trainer.inspect_source_code(hashes)

    def test_reviewed_source_does_not_allow_future_unreviewed_current_trainer(self):
        hashes = trainer.code_signature()
        hashes['train_sam_refine32.py'] = next(iter(trainer.LEGACY_BASE_TRAINERS))
        real = trainer.file_hash
        with patch.dict(namespace, file_hash=lambda p: 'future' if p.name == 'train_sam_refine32.py' else real(p)):
            with self.assertRaisesRegex(ValueError, 'train_sam_refine32.py'):
                trainer.inspect_source_code(hashes)


if __name__ == '__main__':
    unittest.main()
