import json
import os
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch

import download_artifacts as downloader


class ArtifactDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = json.loads((downloader.ROOT / "artifacts.json").read_text())
        self.write_manifest()
        self.hub = Mock()
        self.tiktoken = Mock()
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {}, clear=True).start()
        patch.dict(sys.modules, {"huggingface_hub": self.hub, "tiktoken": self.tiktoken}).start()
        self.urlopen = patch.object(urllib.request, "urlopen", side_effect=AssertionError("Network is forbidden in tests")).start()

    def write_manifest(self):
        (self.root / "artifacts.json").write_text(json.dumps(self.manifest))

    def assert_no_downloads(self):
        self.hub.snapshot_download.assert_not_called()
        self.tiktoken.get_encoding.assert_not_called()
        self.urlopen.assert_not_called()

    def test_default_operations_keep_all_four_pinned_downloads(self):
        downloader.download_artifacts(self.root)
        calls = self.hub.snapshot_download.call_args_list
        self.assertEqual([call.args[0] for call in calls], [self.manifest[key]["model"] for key in ("llmlingua2", "sot", "embed", "verify")])
        for key, call in zip(("llmlingua2", "sot", "embed", "verify"), calls):
            self.assertEqual(call.kwargs, {"revision": self.manifest[key]["revision"], "local_dir": self.root / "models" / key})
        self.assertEqual(os.environ["TIKTOKEN_CACHE_DIR"], str(self.root / "models" / "tiktoken"))
        self.tiktoken.get_encoding.assert_called_once_with("cl100k_base")
        self.urlopen.assert_not_called()

    def test_whitespace_empty_entries_and_duplicates_are_normalized(self):
        os.environ["OPTIMIZATION_OPERATIONS"] = " embed, verify , embed, ,\n"
        downloader.download_artifacts(self.root)
        self.assertEqual([call.args[0] for call in self.hub.snapshot_download.call_args_list], [self.manifest[key]["model"] for key in ("embed", "verify")])
        self.tiktoken.get_encoding.assert_not_called()
        self.urlopen.assert_not_called()

    def test_unknown_names_fail_clearly_before_any_downloads(self):
        os.environ["OPTIMIZATION_OPERATIONS"] = "embed, unsupported-model, other"
        with self.assertRaisesRegex(RuntimeError, "Unknown OPTIMIZATION_OPERATIONS: unsupported-model, other.*Supported operations"):
            downloader.download_artifacts(self.root)
        self.assert_no_downloads()

    def test_explicit_empty_configuration_disables_all_downloads(self):
        os.environ["OPTIMIZATION_OPERATIONS"] = " , \n, "
        downloader.download_artifacts(self.root)
        self.assert_no_downloads()
        self.assertNotIn("TIKTOKEN_CACHE_DIR", os.environ)

    def test_removed_operation_blocks_the_entire_selection_before_download(self):
        os.environ["OPTIMIZATION_OPERATIONS"] = "embed, provence"
        with self.assertRaisesRegex(RuntimeError, "Unknown OPTIMIZATION_OPERATIONS: provence"):
            downloader.download_artifacts(self.root)
        self.assert_no_downloads()

    def test_all_selected_artifacts_require_pinned_revisions_before_download(self):
        os.environ["OPTIMIZATION_OPERATIONS"] = "embed, verify"
        for invalid in ("main", None, 123, "a" * 39):
            with self.subTest(revision=invalid):
                self.manifest["verify"]["revision"] = invalid
                self.write_manifest()
                with self.assertRaisesRegex(RuntimeError, "Artifact 'verify'.*pinned 40-character revision"):
                    downloader.download_artifacts(self.root)
                self.assert_no_downloads()

    def test_malformed_manifest_or_missing_model_fails_before_download(self):
        os.environ["OPTIMIZATION_OPERATIONS"] = "embed"
        for invalid in ({"embed": None}, {"embed": {"revision": "a" * 40}}, []):
            with self.subTest(manifest=invalid):
                self.manifest = invalid
                self.write_manifest()
                with self.assertRaises(RuntimeError):
                    downloader.download_artifacts(self.root)
                self.assert_no_downloads()
if __name__ == "__main__":
    unittest.main()
