import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fsdp_research_resources as resources


class ResourceDiagnosticsTests(unittest.TestCase):
    def test_v2_reports_ancestor_limit_and_counter_changes_without_claiming_a_victim(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); proc = root / 'proc'; task = proc / '123'
            task.mkdir(parents=True); (proc / 'self').mkdir()
            mount = root / 'memory mount'; group = mount / 'account/job'; group.mkdir(parents=True)
            (task / 'cgroup').write_text('0::/account/job\n')
            escaped = str(mount).replace(' ', r'\040')
            (proc / 'self/mountinfo').write_text(f'1 0 0:1 / {escaped} rw - cgroup2 cgroup rw\n')
            (task / 'status').write_text('Name:\tpython\nVmRSS:\t1024 kB\nSecret:\tignored\n')
            (task / 'limits').write_text('Max address space unlimited unlimited bytes\n')
            (proc / 'meminfo').write_text('MemTotal: 2000000 kB\nMemAvailable: 1600000 kB\n')
            (group / 'memory.events').write_text('oom 0\noom_kill 0\n')
            (group.parent / 'memory.max').write_text(str(16 * 2**30))
            before = resources.snapshot(123, proc)
            self.assertEqual(before['process']['VmRSS'], '1024 kB')
            self.assertNotIn('Secret', before['process'])
            groups = before['cgroup']['groups']
            self.assertEqual(len(groups), 3)
            self.assertEqual(groups[1]['files']['memory.max']['value'], str(16 * 2**30))
            (group / 'memory.events').write_text('oom 1\noom_kill 1\n')
            after = resources.snapshot(123, proc)
            deltas = resources.event_deltas(before, after)
            self.assertEqual({row['counter']: row['delta'] for row in deltas}, {'oom': 1, 'oom_kill': 1})
            self.assertIn('do not identify an OOM victim', after['cgroup']['scope'])

    def test_v1_memory_controller_and_mount_subtree_are_resolved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); proc = root / 'proc'; task = proc / '123'
            task.mkdir(parents=True); (proc / 'self').mkdir()
            mount = root / 'memory'; group = mount / 'job'; group.mkdir(parents=True)
            (task / 'cgroup').write_text('5:memory:/account/job\n')
            (proc / 'self/mountinfo').write_text(f'1 0 0:1 /account {mount} rw - cgroup cgroup rw,memory\n')
            (group / 'memory.limit_in_bytes').write_text('1000')
            (group / 'memory.failcnt').write_text('7')
            report = resources.snapshot(123, proc)
            self.assertEqual(report['cgroup']['groups'][0]['version'], 1)
            self.assertEqual(report['cgroup']['groups'][0]['files']['memory.limit_in_bytes']['value'], '1000')

    def test_missing_proc_is_reported_without_failing_the_workflow(self):
        with tempfile.TemporaryDirectory() as folder:
            report = resources.snapshot(123, Path(folder))
            self.assertEqual(report['process_read_error'], 'FileNotFoundError')
            self.assertEqual(report['cgroup']['groups'], [])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Requires Linux procfs')
    def test_real_unprivileged_self_snapshot(self):
        report = resources.snapshot()
        self.assertEqual(report['pid'], os.getpid())
        self.assertIsNone(report['process_read_error'])
        self.assertIn('VmRSS', report['process'])


if __name__ == '__main__':
    unittest.main()
