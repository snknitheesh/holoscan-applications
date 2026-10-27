# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of prepare_model.py without the SDK, a GPU, Ultralytics or the network.

A fake `yolo` command writes fixture ONNX models, downloads come from local
file:// URLs, and the fixtures are encoded here as minimal ONNX protobuf.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prepare_model

COCO_NAMES = "{0: 'person', 1: 'bicycle', 2: 'traffic light'}"


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _field(number: int, payload: bytes | int) -> bytes:
    if isinstance(payload, int):
        return _varint(number << 3) + _varint(payload)
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _value_info(name: str, dims: list[int | str]) -> bytes:
    shape = b"".join(
        _field(1, _field(2, d.encode()) if isinstance(d, str) else _field(1, d)) for d in dims
    )
    tensor_type = _field(1, 1) + _field(2, shape)
    return _field(1, name.encode()) + _field(2, _field(1, tensor_type))


def onnx_model(
    input_dims: list[int | str],
    output_dims: list[int | str],
    names: str | None = COCO_NAMES,
    initializer: str | None = None,
) -> bytes:
    """A ModelProto with one graph input and output, and optional metadata."""
    graph = _field(11, _value_info("images", input_dims))
    if initializer:
        # An initializer listed among the inputs, as older exporters do.
        graph += _field(5, _field(8, initializer.encode()) + _field(9, b"\0" * 4))
        graph += _field(11, _value_info(initializer, [1]))
    graph += _field(12, _value_info("output0", output_dims))
    model = _field(1, 9) + _field(7, graph)
    if names is not None:
        model += _field(14, _field(1, b"names") + _field(2, names.encode()))
        model += _field(14, _field(1, b"task") + _field(2, b"detect"))
    return model


def fake_yolo(directory: Path, model: bytes, exit_status: int = 0) -> Path:
    """A `yolo` command that writes `model` beside the weights it is given."""
    fixture = directory / "fixture.onnx"
    fixture.write_bytes(model)
    script = directory / "yolo"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import pathlib, shutil, sys
            args = dict(a.split("=", 1) for a in sys.argv[2:])
            assert sys.argv[1] == "export", sys.argv
            assert args["format"] == "onnx" and args["nms"] == "False", args
            weights = pathlib.Path(args["model"])
            assert weights.is_file(), weights
            import os
            pathlib.Path({str(directory / "autoinstall.txt")!r}).write_text(
                os.environ.get("YOLO_AUTOINSTALL", "unset"))
            shutil.copyfile({str(fixture)!r}, weights.with_suffix(".onnx"))
            sys.exit({exit_status})
            """
        )
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


class PrepareModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.output = self.root / "data"
        self.weights = self.root / "custom.pt"
        self.weights.write_bytes(b"weights")
        self.addCleanup(self._temporary.cleanup)

    def run_main(self, *arguments: str) -> tuple[int, str]:
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            status = prepare_model.main(list(arguments))
        return status, stderr.getvalue()

    def test_reads_model_inputs_outputs_and_names(self) -> None:
        path = self.root / "model.onnx"
        path.write_bytes(onnx_model([1, 3, 640, 640], [1, 300, 6], initializer="const"))
        info = prepare_model.read_model_info(path)
        self.assertEqual(info.input_shape, [1, 3, 640, 640])
        self.assertEqual(info.output_shape, [1, 300, 6])
        self.assertEqual(info.names, ["person", "bicycle", "traffic light"])

    def test_symbolic_dimensions_read_as_minus_one(self) -> None:
        path = self.root / "model.onnx"
        path.write_bytes(onnx_model(["batch", 3, 640, 640], [1, 300, 6]))
        self.assertEqual(prepare_model.read_model_info(path).input_shape, [-1, 3, 640, 640])

    def test_rejects_malformed_files(self) -> None:
        for content in (b"", b"\xff\xff\xff", onnx_model([1, 3, 640, 640], [1, 300, 6])[:-3]):
            path = self.root / "bad.onnx"
            path.write_bytes(content)
            with self.assertRaises(prepare_model.PreparationError):
                prepare_model.read_model_info(path)

    def test_check_model(self) -> None:
        good = prepare_model.ModelInfo([1, 3, 640, 640], [1, 300, 6], ["a"])
        prepare_model.check_model(good, 640)
        with self.assertRaisesRegex(prepare_model.PreparationError, r"not \[1, 3, 320, 320\]"):
            prepare_model.check_model(good, 320)
        one_to_many = prepare_model.ModelInfo([1, 3, 640, 640], [1, 84, 8400], ["a"])
        with self.assertRaisesRegex(prepare_model.PreparationError, "nms=False"):
            prepare_model.check_model(one_to_many, 640)

    @mock.patch.dict(os.environ)
    def test_exports_local_weights(self) -> None:
        os.environ.pop("YOLO_AUTOINSTALL", None)
        yolo = fake_yolo(self.root, onnx_model([1, 3, 320, 320], [1, 300, 6]))
        stale = self.output / "custom.GPU.12.0.1.trt.10.16.1.1.cfg0.engine.fp32"
        self.output.mkdir()
        stale.write_bytes(b"old engine")
        status, error = self.run_main(
            "--output",
            str(self.output),
            "--weights",
            str(self.weights),
            "--imgsz",
            "320",
            "--yolo",
            str(yolo),
            "--no-samples",
        )
        self.assertEqual(status, 0, error)
        self.assertTrue((self.output / "custom.onnx").is_file())
        self.assertEqual(
            (self.output / "custom.labels.txt").read_text(), "person\nbicycle\ntraffic light\n"
        )
        # The exporter was told not to install packages.
        self.assertEqual((self.root / "autoinstall.txt").read_text(), "false")
        # The engine cached for the previous export is gone, and nothing is
        # left of the work directory.
        self.assertFalse(stale.exists())
        self.assertEqual(
            sorted(p.name for p in self.output.iterdir()), ["custom.labels.txt", "custom.onnx"]
        )

    def test_existing_model_is_kept_unless_forced(self) -> None:
        yolo = fake_yolo(self.root, onnx_model([1, 3, 640, 640], [1, 300, 6]))
        arguments = ["--output", str(self.output), "--weights", str(self.weights)]
        arguments += ["--yolo", str(yolo), "--no-samples"]
        self.assertEqual(self.run_main(*arguments)[0], 0)
        onnx = self.output / "custom.onnx"
        onnx.write_bytes(onnx_model([1, 3, 640, 640], [1, 100, 6]))
        self.assertEqual(self.run_main(*arguments)[0], 0)
        self.assertEqual(prepare_model.read_model_info(onnx).output_shape, [1, 100, 6])
        self.assertEqual(self.run_main(*arguments, "--force")[0], 0)
        self.assertEqual(prepare_model.read_model_info(onnx).output_shape, [1, 300, 6])

    def test_rejects_exports_the_application_cannot_run(self) -> None:
        cases = {
            "nms=False": onnx_model([1, 3, 640, 640], [1, 84, 8400]),
            "no class names": onnx_model([1, 3, 640, 640], [1, 300, 6], names=None),
            "not numbered": onnx_model([1, 3, 640, 640], [1, 300, 6], names="{1: 'a'}"),
        }
        for message, model in cases.items():
            with self.subTest(message):
                yolo = fake_yolo(self.root, model)
                status, error = self.run_main(
                    "--output", str(self.output), "--weights", str(self.weights),
                    "--yolo", str(yolo), "--no-samples",
                )  # fmt: skip
                self.assertEqual(status, 1)
                self.assertIn(message, error)
                self.assertFalse((self.output / "custom.onnx").exists())

    @mock.patch.dict(os.environ, {"YOLO_AUTOINSTALL": "true"})
    def test_an_explicit_autoinstall_setting_is_kept(self) -> None:
        yolo = fake_yolo(self.root, onnx_model([1, 3, 640, 640], [1, 300, 6]))
        status, error = self.run_main(
            "--output", str(self.output), "--weights", str(self.weights),
            "--yolo", str(yolo), "--no-samples",
        )  # fmt: skip
        self.assertEqual(status, 0, error)
        self.assertEqual((self.root / "autoinstall.txt").read_text(), "true")

    def test_reports_a_failed_export(self) -> None:
        yolo = fake_yolo(self.root, onnx_model([1, 3, 640, 640], [1, 300, 6]), exit_status=3)
        status, error = self.run_main(
            "--output", str(self.output), "--weights", str(self.weights),
            "--yolo", str(yolo), "--no-samples",
        )  # fmt: skip
        self.assertEqual(status, 1)
        self.assertIn("exit status 3", error)

    def test_missing_yolo_command_says_how_to_install_it(self) -> None:
        status, error = self.run_main(
            "--output", str(self.output), "--yolo", str(self.root / "missing" / "yolo"),
            "--no-samples",
        )  # fmt: skip
        self.assertEqual(status, 1)
        self.assertIn("AGPL-3.0", error)
        self.assertIn("--yolo", error)

    def test_rejects_bad_input_size(self) -> None:
        for size in ("100", "0", "4128"):
            status, error = self.run_main("--output", str(self.output), "--imgsz", size)
            self.assertEqual(status, 1)
            self.assertIn("multiple of 32", error)

    def test_pinned_weights_are_downloaded_and_checked(self) -> None:
        source = self.root / "served"
        source.mkdir()
        (source / "yolo26n.pt").write_bytes(b"pinned weights")
        digest = hashlib.sha256(b"pinned weights").hexdigest()
        yolo = fake_yolo(self.root, onnx_model([1, 3, 640, 640], [1, 300, 6]))
        url = source.as_uri() + "/{name}.pt"
        with (
            mock.patch.object(prepare_model, "WEIGHTS_URL", url),
            mock.patch.dict(prepare_model.WEIGHTS_SHA256, {"yolo26n": digest}),
        ):
            status, error = self.run_main(
                "--output", str(self.output), "--yolo", str(yolo), "--no-samples"
            )
        self.assertEqual(status, 0, error)
        self.assertTrue((self.output / "yolo26n.onnx").is_file())
        with (
            mock.patch.object(prepare_model, "WEIGHTS_URL", url),
            mock.patch.dict(prepare_model.WEIGHTS_SHA256, {"yolo26n": "0" * 64}),
        ):
            status, error = self.run_main(
                "--output", str(self.output), "--yolo", str(yolo), "--no-samples", "--force"
            )
        self.assertEqual(status, 1)
        self.assertIn("does not match the pinned", error)

    def test_samples_are_downloaded_checked_and_reused(self) -> None:
        served = self.root / "sample.jpg"
        served.write_bytes(b"jpeg bytes")
        digest = hashlib.sha256(b"jpeg bytes").hexdigest()
        samples = {"sample.jpg": (served.as_uri(), digest)}
        with mock.patch.object(prepare_model, "SAMPLE_IMAGES", samples):
            prepare_model.prepare_samples(self.output)
            image = self.output / "images" / "sample.jpg"
            self.assertEqual(image.read_bytes(), b"jpeg bytes")
            # A good copy is not downloaded again.
            served.unlink()
            prepare_model.prepare_samples(self.output)
            # A corrupt one is, and fails without a source.
            image.write_bytes(b"corrupt")
            with self.assertRaises(prepare_model.PreparationError):
                prepare_model.prepare_samples(self.output)
        bad = {"sample.jpg": (self.weights.as_uri(), digest)}
        with (
            mock.patch.object(prepare_model, "SAMPLE_IMAGES", bad),
            self.assertRaisesRegex(prepare_model.PreparationError, "does not match"),
        ):
            prepare_model.prepare_samples(self.root / "other")
        self.assertEqual(os.listdir(self.root / "other" / "images"), [])


if __name__ == "__main__":
    unittest.main()
