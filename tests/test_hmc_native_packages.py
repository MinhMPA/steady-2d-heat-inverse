"""Complete native comparisons with real files and package ownership records."""

from importlib.machinery import ModuleSpec
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import native_provenance as native
from tests import test_hmc_native_aliases as aliases


class NativePackageTests(unittest.TestCase):
    setUp = aliases.NativeAliasTests.setUp
    write = aliases.NativeAliasTests.write

    def probe(self, *args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout="S2DHI_NATIVE_JSON="
            + json.dumps(
                {
                    "imports": self.imports,
                    "images": self.fresh,
                    "packages": getattr(
                        self, "fresh_packages", {"packages": {}, "files": {}}
                    ),
                }
            ),
        )

    def package(self, name, relative, *, prefix=None, module=None):
        site = (prefix or self.prefix) / "lib/python3.12/site-packages"
        origin = self.write(site / name / "__init__.py", b"# controlled package")
        binary = self.write(site / name / relative, name.encode())
        info = site / (name + "-1.dist-info")
        info.mkdir(exist_ok=True)
        (info / "METADATA").write_text(f"Name: {name}\nVersion: 1\n")
        (info / "RECORD").write_text(f"{name}/__init__.py,,\n{name}/{relative},,\n")
        package = ModuleType(name)
        package.__file__ = str(origin)
        package.__path__ = [str(origin.parent)]
        package.__spec__ = ModuleSpec(name, None, origin=str(origin), is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        self.enterContext(patch.dict(sys.modules, {name: package}))
        if module:
            extension = ModuleType(module)
            extension.__file__ = str(binary)
            extension.__spec__ = ModuleSpec(module, None, origin=str(binary))
            self.enterContext(patch.dict(sys.modules, {module: extension}))
        return binary, {
            "root": str(origin.parent),
            "origin": str(origin),
            "owner": str(info / "METADATA"),
        }

    def numerical_extension(self):
        binary, package = self.package(
            "scipy", "spatial/_qhull.so", module="scipy.spatial._qhull"
        )
        self.fresh_packages = {
            "packages": {"scipy": package},
            "files": {
                str(binary): {
                    "package": "scipy",
                    "relative": "spatial/_qhull.so",
                    "modules": ["scipy.spatial._qhull"],
                    "valid": True,
                }
            },
        }
        self.parent = self.fresh = [str(self.local), str(binary)]
        return binary

    def test_distinct_python_modules_with_same_basename_preserve_identity(self):
        self.numerical_extension()
        expected = native.native_build_identity()
        other, _ = self.package("matplotlib", "_qhull.so", module="matplotlib._qhull")
        self.parent = [*self.parent, str(other)]
        diagnostics = {}
        self.assertEqual(
            expected, native.native_build_identity(image_diagnostics=diagnostics)
        )
        self.assertIn(
            str(other), [row["path"] for row in diagnostics["excluded_parent"]]
        )
        self.parent.reverse()
        self.assertEqual(expected, native.native_build_identity())

    def test_private_helper_requires_unchanged_original_runtime(self):
        runtime = self.write(self.prefix / "lib/liblzma.5.dylib", b"global runtime")
        self.parent = self.fresh = [str(self.local), str(runtime)]
        expected = native.native_build_identity()
        helper, _ = self.package("PIL", ".dylibs/liblzma.5.dylib")
        self.parent = [str(self.local), str(runtime), str(helper)]
        self.assertEqual(expected, native.native_build_identity())
        self.parent = [str(self.local), str(helper)]
        with self.assertRaisesRegex(RuntimeError, "runtime|disagree"):
            native.native_build_identity()
        replacement = self.write(
            self.root / "other/lib/liblzma.5.dylib", runtime.read_bytes()
        )
        self.parent = [str(self.local), str(replacement), str(helper)]
        with self.assertRaisesRegex(RuntimeError, "runtime|disagree|Ambiguous"):
            native.native_build_identity()

    def test_private_helper_without_unique_ownership_cannot_exempt_collision(self):
        runtime = self.write(self.prefix / "lib/liblzma.5.dylib")
        self.parent = self.fresh = [str(self.local), str(runtime)]
        helper, _ = self.package("PIL", ".dylibs/liblzma.5.dylib")
        info = helper.parents[2] / "unrelated-1.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Name: unrelated\nVersion: 1\n")
        (info / "RECORD").write_text("PIL/.dylibs/liblzma.5.dylib,,\n")
        self.parent = [*self.parent, str(helper)]
        with self.assertRaises(RuntimeError):
            native.native_build_identity()

    def test_changed_ownership_records_are_reinspected(self):
        runtime = self.write(self.prefix / "lib/liblzma.5.dylib")
        self.parent = self.fresh = [str(self.local), str(runtime)]
        helper, package = self.package("PIL", ".dylibs/liblzma.5.dylib")
        info = helper.parents[2] / "unrelated-1.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Name: unrelated\nVersion: 1\n")
        unrelated_record = info / "RECORD"
        unrelated_record.write_text("unrelated/ordinary.py,,\n")
        self.parent = [*self.parent, str(helper)]
        diagnostics = {}
        expected = native.native_build_identity(image_diagnostics=diagnostics)
        self.assertIn(
            str(helper), [row["path"] for row in diagnostics["excluded_parent"]]
        )

        # An already present distribution now claims the same helper. Its new
        # ownership record must invalidate the previously proved exception.
        unrelated_record.write_text("PIL/.dylibs/liblzma.5.dylib,,\n")
        with self.assertRaisesRegex(RuntimeError, "Ambiguous|disagree"):
            native.native_build_identity()
        unrelated_record.write_text("unrelated/ordinary.py,,\n")
        self.assertEqual(expected, native.native_build_identity())

        # Revoking the imported package origin also removes the positive proof.
        package_record = Path(package["owner"]).with_name("RECORD")
        original_record = package_record.read_text()
        package_record.write_text("PIL/.dylibs/liblzma.5.dylib,,\n")
        with self.assertRaisesRegex(RuntimeError, "Ambiguous|disagree"):
            native.native_build_identity()
        package_record.write_text(original_record)
        self.assertEqual(expected, native.native_build_identity())

    def test_hidden_core_build_rejected_even_with_original_present(self):
        helper, _ = self.package("PIL", ".dylibs/" + self.local.name)
        self.parent = [*self.parent, str(helper)]
        with self.assertRaisesRegex(RuntimeError, "Ambiguous|disagree"):
            native.native_build_identity()

    def test_extra_foreign_numerical_image_rejected_with_original_present(self):
        original = self.numerical_extension()
        for contents in [b"foreign build", original.read_bytes()]:
            with self.subTest(contents=contents):
                foreign = self.write(
                    self.sibling / original.relative_to(self.prefix), contents
                )
                self.parent = [*self.fresh, str(foreign)]
                with self.assertRaises(RuntimeError):
                    native.native_build_identity()

    def test_foreign_numerical_extra_without_fresh_basename_rejected(self):
        self.numerical_extension()
        foreign = self.write(
            self.sibling / "lib/python3.12/site-packages/scipy/fftpack/convolve.so"
        )
        self.parent = [*self.fresh, str(foreign)]
        with self.assertRaises(RuntimeError):
            native.native_build_identity()

    def test_numerical_module_under_misleading_package_path_rejected(self):
        self.numerical_extension()
        other, _ = self.package("matplotlib", "different.so", module="scipy.hidden")
        self.parent = [*self.fresh, str(other)]
        with self.assertRaises(RuntimeError):
            native.native_build_identity()

    def test_extra_same_installation_numerical_module_is_diagnostic_only(self):
        self.numerical_extension()
        expected = native.native_build_identity()
        extra, _ = self.package(
            "scipy", "fftpack/convolve.so", module="scipy.fftpack.convolve"
        )
        # Keep ownership of both the reference and extra file.
        record = extra.parents[2] / "scipy-1.dist-info/RECORD"
        with record.open("a") as stream:
            stream.write("scipy/spatial/_qhull.so,,\n")
        self.parent = [*self.fresh, str(extra)]
        diagnostics = {}
        self.assertEqual(
            expected, native.native_build_identity(image_diagnostics=diagnostics)
        )
        self.assertIn(
            str(extra), [row["path"] for row in diagnostics["excluded_parent"]]
        )

    def test_samefile_numerical_alias_preserves_identity(self):
        original = self.numerical_extension()
        expected = native.native_build_identity()
        alias = self.sibling / original.relative_to(self.prefix)
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.hardlink_to(original)
        self.parent = [str(self.local), str(alias)]
        self.assertEqual(expected, native.native_build_identity())

    def test_missing_fresh_numerical_extension_rejected(self):
        original = self.numerical_extension()
        self.fresh = [str(self.local)]
        # An existing declared counterpart must not disappear from the image list.
        with self.assertRaises(RuntimeError):
            native.native_build_identity()

    def test_pybind_aliases_without_spec_preserve_verified_binary(self):
        original = self.numerical_extension()
        expected = native.native_build_identity()
        alias = ModuleType("scipy.spatial._qhull.child")
        alias.__file__ = str(original)
        with patch.dict(sys.modules, {alias.__name__: alias}):
            self.assertEqual(expected, native.native_build_identity())

    def test_top_level_alias_uses_declared_module_origin(self):
        self.numerical_extension()
        expected = native.native_build_identity()
        with patch.dict(sys.modules, {"_qhull": sys.modules["scipy.spatial._qhull"]}):
            self.assertEqual(expected, native.native_build_identity())

    def test_vendored_extension_uses_registered_spec_origin(self):
        self.numerical_extension()
        expected = native.native_build_identity()
        sys.modules["scipy.spatial._qhull"].__name__ = "vendored._qhull"
        self.assertEqual(expected, native.native_build_identity())

    def test_flat_namespace_is_recorded_and_cannot_gain_helper_exemption(self):
        runtime = self.write(self.prefix / "lib/liblzma.5.dylib")
        self.parent = self.fresh = [str(self.local), str(runtime)]
        with patch.dict(native.os.environ, {"DYLD_FORCE_FLAT_NAMESPACE": "1"}):
            identity = native.native_build_identity()
            self.assertEqual(
                identity["loader_inputs"].get("DYLD_FORCE_FLAT_NAMESPACE"), "1"
            )
            helper, _ = self.package("PIL", ".dylibs/liblzma.5.dylib")
            self.parent = [*self.parent, str(helper)]
            with self.assertRaises(RuntimeError):
                native.native_build_identity()

    def module_proxy(self, module):
        # CFFI library namespaces can claim ModuleType through __class__ while
        # failing the real ModuleType descriptor used to inspect stored fields.
        class ModuleProxy:
            @property
            def __class__(self):
                return ModuleType

        proxy = ModuleProxy()
        proxy.__dict__.update(ModuleType.__getattribute__(module, "__dict__"))
        self.assertTrue(isinstance(proxy, ModuleType))
        self.assertFalse(issubclass(type(proxy), ModuleType))
        return proxy

    def test_nonmodule_namespace_does_not_block_or_hide_numerical_comparison(self):
        self.numerical_extension()
        expected = native.native_build_identity()
        proxy = self.module_proxy(ModuleType("controlled_cffi_namespace"))
        with patch.dict(sys.modules, {"controlled_cffi_namespace": proxy}):
            self.assertEqual(expected, native.native_build_identity())
            foreign = self.write(self.root / "foreign" / "lib" / self.local.name)
            self.parent = [*self.parent, str(foreign)]
            with self.assertRaisesRegex(RuntimeError, "Ambiguous|disagree"):
                native.native_build_identity()

    def test_nonmodule_namespace_cannot_prove_private_package_ownership(self):
        runtime = self.write(self.prefix / "lib/liblzma.5.dylib")
        self.parent = self.fresh = [str(self.local), str(runtime)]
        helper, _ = self.package("PIL", ".dylibs/liblzma.5.dylib")
        proxy = self.module_proxy(sys.modules["PIL"])
        self.parent = [*self.parent, str(helper)]
        with patch.dict(sys.modules, {"PIL": proxy}):
            with self.assertRaisesRegex(RuntimeError, "Ambiguous|disagree"):
                native.native_build_identity()

    def test_real_module_subclass_is_inspected_without_lazy_accessors(self):
        self.numerical_extension()
        expected = native.native_build_identity()

        class LazyModule(ModuleType):
            def __getattribute__(self, name):
                if name in {
                    "__class__",
                    "__dict__",
                    "__file__",
                    "__spec__",
                    "__path__",
                }:
                    raise AssertionError("Lazy module accessor must not run.")
                return super().__getattribute__(name)

        lazy = LazyModule("scipy")
        ModuleType.__getattribute__(lazy, "__dict__").update(
            ModuleType.__getattribute__(sys.modules["scipy"], "__dict__")
        )
        with patch.dict(sys.modules, {"scipy": lazy}):
            self.assertEqual(expected, native.native_build_identity())


if __name__ == "__main__":
    unittest.main()
