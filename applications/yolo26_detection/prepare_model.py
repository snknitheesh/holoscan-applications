#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare an end-to-end YOLO26 ONNX model for the yolo26_detection application.

The YOLO26 weights and the Ultralytics exporter are licensed under AGPL-3.0 and
are not part of this repository. This script downloads pinned weights and
checks their SHA-256 digest, runs the separately installed Ultralytics `yolo`
command to export them to ONNX with the NMS-free head, checks the export, and
writes into the output directory:

- <model>.onnx: input float32 [1, 3, S, S], output float32 [1, N, 6];
- <model>.labels.txt: the class names, one per line;
- images/000000039769.jpg: a COCO validation image for the demo and tests.

It never imports Ultralytics; it only runs its command-line tool, with
YOLO_AUTOINSTALL=false unless that is set, so that the exporter installs no
packages itself.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}.pt"
WEIGHTS_SHA256 = {
    "yolo26n": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    "yolo26s": "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b",
    "yolo26m": "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7",
    "yolo26l": "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26",
    "yolo26x": "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92",
}
SAMPLE_IMAGES = {
    "000000039769.jpg": (
        "https://s3.amazonaws.com/images.cocodataset.org/val2017/000000039769.jpg",
        "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e",
    ),
}
DOWNLOAD_TIMEOUT_SECONDS = 120


class PreparationError(RuntimeError):
    """A user-facing model preparation failure."""


@dataclass(frozen=True)
class ModelInfo:
    """What the application needs to know about an exported model."""

    input_shape: list[int]
    output_shape: list[int]
    names: list[str]


# A minimal protobuf reader, enough to find the inputs, outputs and metadata of
# an ONNX ModelProto without the onnx package.


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data) or shift > 63:
            raise PreparationError("malformed ONNX file: bad varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def _fields(data: bytes) -> list[tuple[int, int, int | bytes]]:
    """Return (field number, wire type, value) for each field of a message."""
    fields = []
    offset = 0
    while offset < len(data):
        key, offset = _varint(data, offset)
        number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, offset = _varint(data, offset)
        elif wire_type == 1:
            value, offset = data[offset : offset + 8], offset + 8
        elif wire_type == 2:
            length, offset = _varint(data, offset)
            if offset + length > len(data):
                raise PreparationError("malformed ONNX file: truncated field")
            value, offset = data[offset : offset + length], offset + length
        elif wire_type == 5:
            value, offset = data[offset : offset + 4], offset + 4
        else:
            raise PreparationError(f"malformed ONNX file: wire type {wire_type}")
        fields.append((number, wire_type, value))
    if offset != len(data):
        raise PreparationError("malformed ONNX file: truncated message")
    return fields


def _messages(data: bytes, number: int) -> list[bytes]:
    return [v for n, w, v in _fields(data) if n == number and w == 2 and isinstance(v, bytes)]


def _string(data: bytes, number: int) -> str:
    values = _messages(data, number)
    return values[0].decode("utf-8") if values else ""


def _shape(value_info: bytes) -> list[int]:
    """The dimensions of a tensor ValueInfoProto; -1 for a symbolic one."""
    dims = []
    for type_proto in _messages(value_info, 2):
        for tensor_type in _messages(type_proto, 1):
            for shape in _messages(tensor_type, 2):
                for dim in _messages(shape, 1):
                    values = [v for n, w, v in _fields(dim) if n == 1 and w == 0]
                    dims.append(int(values[0]) if values else -1)
    return dims


def read_model_info(path: Path) -> ModelInfo:
    """Read the graph input, output and class names of an ONNX model."""
    try:
        data = path.read_bytes()
    except OSError as error:
        raise PreparationError(f"{path}: cannot read: {error}") from error
    graphs = _messages(data, 7)
    if len(graphs) != 1:
        raise PreparationError(f"{path}: not an ONNX model")
    graph = graphs[0]
    initializers = {_string(tensor, 8) for tensor in _messages(graph, 5)}
    inputs = [v for v in _messages(graph, 11) if _string(v, 1) not in initializers]
    outputs = _messages(graph, 12)
    if len(inputs) != 1 or len(outputs) != 1:
        raise PreparationError(f"{path}: expected one input and one output")
    metadata = {}
    for entry in _messages(data, 14):
        metadata[_string(entry, 1)] = _string(entry, 2)
    names: list[str] = []
    if "names" in metadata:
        try:
            parsed = ast.literal_eval(metadata["names"])
        except (ValueError, SyntaxError) as error:
            raise PreparationError(f"{path}: unreadable class names") from error
        if not isinstance(parsed, dict) or sorted(parsed) != list(range(len(parsed))):
            raise PreparationError(f"{path}: class names are not numbered 0..N-1")
        names = [str(parsed[i]) for i in range(len(parsed))]
    return ModelInfo(_shape(inputs[0]), _shape(outputs[0]), names)


def check_model(info: ModelInfo, imgsz: int) -> None:
    """Reject a model the application cannot run."""
    if info.input_shape != [1, 3, imgsz, imgsz]:
        raise PreparationError(f"the model takes {info.input_shape}, not [1, 3, {imgsz}, {imgsz}]")
    shape = info.output_shape
    if len(shape) != 3 or shape[0] != 1 or shape[1] < 1 or shape[2] != 6:
        raise PreparationError(
            f"the model gives {shape}, not end-to-end detections [1, N, 6]; "
            "export it with nms=False"
        )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, sha256: str, destination: Path) -> None:
    """Download url to destination, which appears only when its digest matches."""
    partial = destination.with_name(destination.name + ".partial")
    try:
        with (
            urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            partial.open("wb") as file,
        ):
            shutil.copyfileobj(response, file)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise PreparationError(f"cannot download {url}: {error}") from error
    actual = sha256_of(partial)
    if actual != sha256:
        partial.unlink(missing_ok=True)
        raise PreparationError(f"{url}: SHA-256 {actual} does not match the pinned {sha256}")
    os.replace(partial, destination)


def export(yolo: str, weights: Path, imgsz: int, workdir: Path) -> Path:
    """Export weights to ONNX with the NMS-free head; return the ONNX path."""
    command = [yolo, "export", f"model={weights}", "format=onnx", f"imgsz={imgsz}", "nms=False"]
    # Unless asked to, the exporter installs no packages of its own accord:
    # its dependencies come from the environment the user set up.
    environment = dict(os.environ)
    environment.setdefault("YOLO_AUTOINSTALL", "false")
    print("prepare_model: running " + " ".join(command), flush=True)
    try:
        result = subprocess.run(command, cwd=workdir, env=environment, check=False)
    except OSError as error:
        raise PreparationError(f"cannot run {yolo}: {error}") from error
    if result.returncode != 0:
        raise PreparationError(f"{yolo} export failed with exit status {result.returncode}")
    onnx = weights.with_suffix(".onnx")
    if not onnx.is_file():
        raise PreparationError(f"{yolo} export did not write {onnx.name}")
    return onnx


def prepare_samples(output: Path) -> None:
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    for name, (url, sha256) in SAMPLE_IMAGES.items():
        destination = images / name
        if destination.is_file() and sha256_of(destination) == sha256:
            continue
        print(f"prepare_model: downloading {url}", flush=True)
        download(url, sha256, destination)


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export an end-to-end YOLO26 model to ONNX with the separately installed "
            "Ultralytics command-line tool, and download a sample image."
        )
    )
    parser.add_argument("--output", type=Path, required=True, help="destination directory")
    parser.add_argument(
        "--model",
        choices=sorted(WEIGHTS_SHA256),
        default="yolo26n",
        help="pinned weights to download and export (default: yolo26n)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="export this local .pt file instead, such as a model trained on your classes",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="model input size (default: 640)")
    parser.add_argument(
        "--yolo", default="yolo", help="the Ultralytics command (default: yolo on PATH)"
    )
    parser.add_argument("--no-samples", action="store_true", help="skip the sample image")
    parser.add_argument("--force", action="store_true", help="export again over an existing model")
    return parser.parse_args(arguments)


def prepare(arguments: argparse.Namespace) -> Path:
    """Prepare the model and samples; return the ONNX path."""
    if arguments.imgsz < 32 or arguments.imgsz > 4096 or arguments.imgsz % 32 != 0:
        raise PreparationError("--imgsz must be a multiple of 32 in 32..4096")
    stem = arguments.weights.stem if arguments.weights else arguments.model
    output: Path = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    onnx = output / f"{stem}.onnx"
    labels = output / f"{stem}.labels.txt"
    reuse = onnx.is_file() and labels.is_file() and not arguments.force
    if reuse:
        check_model(read_model_info(onnx), arguments.imgsz)
        print(f"prepare_model: {onnx} exists; pass --force to export it again", flush=True)
    else:
        yolo = shutil.which(arguments.yolo)
        if yolo is None:
            raise PreparationError(
                f"the Ultralytics command '{arguments.yolo}' was not found; install the "
                "ultralytics package (AGPL-3.0) in a separate virtual environment and pass "
                "its yolo command with --yolo"
            )
        with tempfile.TemporaryDirectory(dir=output, prefix=".prepare-") as temporary:
            workdir = Path(temporary)
            weights = workdir / f"{stem}.pt"
            if arguments.weights:
                if not arguments.weights.is_file():
                    raise PreparationError(f"{arguments.weights}: no such file")
                shutil.copyfile(arguments.weights, weights)
            else:
                url = WEIGHTS_URL.format(name=stem)
                print(f"prepare_model: downloading {url}", flush=True)
                download(url, WEIGHTS_SHA256[stem], weights)
            exported = export(yolo, weights, arguments.imgsz, workdir)
            info = read_model_info(exported)
            check_model(info, arguments.imgsz)
            if not info.names:
                raise PreparationError("the export has no class names")
            # A TensorRT engine cached for an earlier export of the same name
            # would otherwise be loaded for this one.
            for stale in output.glob(f"{stem}.*.engine.*"):
                stale.unlink()
            os.replace(exported, onnx)
            labels.write_text("\n".join(info.names) + "\n", encoding="utf-8")
        print(f"prepare_model: wrote {onnx} and {labels}", flush=True)
    if not arguments.no_samples:
        prepare_samples(output)
    return onnx


def main(arguments: list[str] | None = None) -> int:
    try:
        prepare(parse_arguments(arguments))
    except PreparationError as error:
        print(f"prepare_model: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
