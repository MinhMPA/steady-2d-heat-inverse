"""Dependency-free checks of native identity using real temporary hardlinks.

Only loader observations and unrelated compiler/package inspection are supplied
by fixtures. File identity, hashing, selection and comparison use the real code.
"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import native_provenance as native


class NativeAliasTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.prefix = self.root / "envs" / "active"
        self.sibling = self.prefix.parent / "sibling"
        self.local = self.write(self.prefix / "lib" / "libpetsc-test.dylib")
        self.alias = self.sibling / self.local.relative_to(self.prefix)
        self.alias.parent.mkdir(parents=True)
        self.alias.hardlink_to(self.local)
        self.fresh = [str(self.local)]
        self.parent = [str(self.local)]
        self.imports = {"numpy": {"status": "loaded", "path": "fixed-import"}}
        self.enterContext(patch.object(native.sys, "prefix", str(self.prefix)))
        self.enterContext(patch.object(native, "_compiler", return_value={"fixed": 1}))
        self.enterContext(patch.object(native, "_sampling_artifacts", return_value={}))
        self.enterContext(patch.object(native, "_loaded_images", lambda: self.parent))
        self.enterContext(patch.object(native.subprocess, "run", self.probe))
        record = self.prefix / "conda-meta" / "controlled.json"
        record.parent.mkdir()
        record.write_text(
            json.dumps(
                {
                    "name": "controlled",
                    "version": "1",
                    "build": "test",
                    "files": [str(self.local.relative_to(self.prefix))],
                }
            )
        )

    def write(self, path, contents=b"controlled numerical binary"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return path

    def probe(self, *args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout="S2DHI_NATIVE_JSON="
            + json.dumps({"imports": self.imports, "images": self.fresh}),
        )

    def test_parent_and_fresh_hardlinks_preserve_full_identity_repeatedly(self):
        self.assertTrue(self.local.samefile(self.alias))
        expected = native.native_build_identity()
        self.assertIn("controlled", expected["conda_builds"])
        for _ in range(3):
            for parent_alias, fresh_alias in [
                (True, False),
                (False, True),
                (True, True),
            ]:
                with self.subTest(parent=parent_alias, fresh=fresh_alias):
                    self.parent = [str(self.alias if parent_alias else self.local)]
                    self.fresh = [str(self.alias if fresh_alias else self.local)]
                    self.assertEqual(expected, native.native_build_identity())

    def test_aliases_normalize_before_prefix_selection(self):
        # This component has neither a numerical basename nor site-packages.
        local = self.write(self.prefix / "lib" / "internal-runtime.dylib")
        alias = self.sibling / local.relative_to(self.prefix)
        alias.hardlink_to(local)
        self.fresh = [str(local)]
        self.parent = [str(local)]
        expected = native.native_build_identity()
        self.fresh = [str(alias)]
        self.assertEqual(expected, native.native_build_identity())

    def test_same_file_aliases_deduplicate_before_sorting(self):
        second = self.write(self.prefix / "lib" / "libmpi-test.dylib")
        self.fresh = [str(second), str(self.local)]
        expected = native.native_build_identity()
        self.fresh = [str(self.alias), str(second), str(self.local), str(self.alias)]
        self.parent = [str(self.alias), str(self.local)]
        actual = native.native_build_identity()
        self.assertEqual(expected, actual)
        self.assertEqual(
            [item["path"] for item in actual["libraries"]],
            sorted([str(self.local), str(second)]),
        )

    def test_different_inode_rejected_even_with_identical_bytes(self):
        for contents in [self.local.read_bytes(), b"different binary"]:
            with self.subTest(contents=contents):
                self.alias.unlink()
                self.alias.write_bytes(contents)
                self.assertFalse(self.local.samefile(self.alias))
                self.fresh = [str(self.alias)]
                with self.assertRaisesRegex(RuntimeError, "disagree"):
                    native.native_build_identity()

    def test_missing_local_counterpart_retains_observed_identity(self):
        self.local.unlink()
        self.parent = self.fresh = [str(self.alias)]
        evidence = native.native_build_identity()
        self.assertEqual(evidence["libraries"][0]["path"], str(self.alias))
        self.assertNotIn("controlled", evidence["conda_builds"])

    def test_outside_sibling_layout_is_not_normalized(self):
        outside = self.root / "elsewhere" / "lib" / self.local.name
        outside.parent.mkdir(parents=True)
        outside.hardlink_to(self.local)
        self.fresh = [str(outside)]
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            native.native_build_identity()

    def test_wrong_relative_suffix_is_not_normalized(self):
        other = self.sibling / "other" / self.local.name
        other.parent.mkdir(parents=True)
        other.hardlink_to(self.local)
        self.fresh = [str(other)]
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            native.native_build_identity()

    def test_fresh_duplicate_basename_rejected_in_either_order(self):
        other = self.write(self.prefix / "other" / self.local.name)
        self.assertFalse(other.samefile(self.local))
        for order in [(other, self.local), (self.local, other)]:
            with self.subTest(order=order):
                self.fresh = [str(path) for path in order]
                self.parent = []
                with self.assertRaisesRegex(RuntimeError, "Ambiguous.*basename"):
                    native.native_build_identity()

    def test_parent_duplicate_basename_rejected_in_either_order(self):
        other = self.write(self.prefix / "other" / self.local.name)
        for order in [(other, self.local), (self.local, other)]:
            with self.subTest(order=order):
                self.parent = [str(path) for path in order]
                with self.assertRaisesRegex(RuntimeError, "Ambiguous.*basename"):
                    native.native_build_identity()

    def test_changed_bytes_at_same_canonical_path_change_identity(self):
        before = native.native_build_identity()
        self.local.write_bytes(b"changed numerical binary")
        self.parent = self.fresh = [str(self.alias)]
        after = native.native_build_identity()
        self.assertEqual(before["libraries"][0]["path"], after["libraries"][0]["path"])
        self.assertNotEqual(
            before["libraries"][0]["sha256"], after["libraries"][0]["sha256"]
        )

    def test_parent_and_fresh_hash_disagreement_still_rejected(self):
        actual_sha = native._sha
        calls = 0

        def changed_observation(path):
            nonlocal calls
            if Path(path) == self.local:
                calls += 1
                if calls == 2:
                    return "f" * 64
            return actual_sha(path)

        with patch.object(native, "_sha", changed_observation):
            with self.assertRaisesRegex(RuntimeError, "disagree"):
                native.native_build_identity()

    def test_alias_bytes_are_verified_at_both_names(self):
        self.fresh = [str(self.alias)]
        actual_sha = native._sha

        def changed_observation(path):
            return "f" * 64 if Path(path) == self.local else actual_sha(path)

        with patch.object(native, "_sha", changed_observation):
            with self.assertRaisesRegex(RuntimeError, "alias changed"):
                native.native_build_identity()

    def test_samefile_proof_is_rechecked_after_hashing(self):
        self.fresh = [str(self.alias)]
        actual_sha = native._sha

        def replace_local_after_hash(path):
            digest = actual_sha(path)
            if Path(path) == self.local:
                contents = self.local.read_bytes()
                self.local.unlink()
                self.local.write_bytes(contents)
            return digest

        with patch.object(native, "_sha", replace_local_after_hash):
            with self.assertRaisesRegex(RuntimeError, "alias changed"):
                native.native_build_identity()
        self.assertFalse(self.local.samefile(self.alias))

    def test_alias_evidence_is_separate_from_identity(self):
        expected = native.native_build_identity()
        self.fresh = self.parent = [str(self.alias)]
        diagnostics = {}
        actual = native.native_build_identity(image_diagnostics=diagnostics)
        self.assertEqual(expected, actual)
        for phase in ["fresh", "parent"]:
            self.assertEqual(
                diagnostics[phase],
                [
                    {
                        "loader_path": str(self.alias),
                        "observed_path": str(self.alias),
                        "canonical_path": str(self.local),
                        "sha256": native._sha(self.alias),
                        "sibling_samefile_verified": True,
                    }
                ],
            )
        self.assertNotIn(str(self.alias), json.dumps(actual))

    def test_symlink_counterpart_cannot_escape_active_prefix(self):
        outside = self.root / "outside" / self.local.name
        outside.parent.mkdir()
        self.local.rename(outside)
        self.local.symlink_to(outside)
        self.fresh = [str(self.alias)]
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            native.native_build_identity()

    def test_hash_change_during_alias_deduplication_rejected(self):
        self.fresh = [str(self.local), str(self.local)]
        actual_sha = native._sha
        calls = 0

        def changed_observation(path):
            nonlocal calls
            calls += 1
            return "f" * 64 if calls == 2 else actual_sha(path)

        with patch.object(native, "_sha", changed_observation):
            with self.assertRaisesRegex(RuntimeError, "bytes changed"):
                native.native_build_identity()


if __name__ == "__main__":
    unittest.main()
