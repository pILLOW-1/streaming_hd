import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RepositoryScopeTests(unittest.TestCase):
    def test_ours_is_readme_only(self):
        self.assertEqual([p.name for p in (ROOT / 'code/ours').iterdir()], ['README.md'])

    def test_imported_research_records_are_hash_bound(self):
        manifest = json.loads((ROOT / 'ara/IMPORT_MANIFEST.json').read_text())
        paths = []
        for entry in manifest['files']:
            p = ROOT / entry['published_path']
            self.assertTrue(p.is_file(), entry['published_path'])
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), entry['published_SHA256'])
            paths.append(entry['published_path'])
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(row['path'].endswith('.lock') for row in manifest['omitted']))
        self.assertTrue(any(p.endswith('trace/exploration_tree.yaml') for p in paths))
        self.assertTrue(any(p.endswith('staging/observations.yaml') for p in paths))

    def test_session_coverage_and_evidence_limits(self):
        summary = json.loads((ROOT/'ara/IMPORT_SUMMARY.json').read_text())
        catalog = json.loads((ROOT/'ara/trace/SESSION_CATALOG.json').read_text())
        self.assertEqual(len(catalog), summary['session_days'])
        for row in catalog:
            self.assertTrue((ROOT/row['path']).is_file())
        refs = json.loads((ROOT/'ara/evidence/EXTERNAL_REFERENCES.json').read_text())
        self.assertEqual(len(refs), summary['external_references'])
        self.assertEqual(sum(row['status']=='not_found_in_original_workspace' for row in refs), summary['references_not_found'])


if __name__ == '__main__':
    unittest.main()
