from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import tarfile
import zipfile
import gzip

from ssl_asr.utils.io import md5_file, sha256_file, write_json


def _safe_extract_tar(tf: tarfile.TarFile, *, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    for m in tf.getmembers():
        p = (out_dir / m.name).resolve()
        if not str(p).startswith(str(out_dir) + os.sep):
            raise RuntimeError(f"Unsafe path in tar archive: {m.name}")
    tf.extractall(path=out_dir)


def _safe_extract_zip(zf: zipfile.ZipFile, *, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    for n in zf.namelist():
        p = (out_dir / n).resolve()
        if not str(p).startswith(str(out_dir) + os.sep):
            raise RuntimeError(f"Unsafe path in zip archive: {n}")
    zf.extractall(path=out_dir)


@dataclass(frozen=True)
class Resource:
    name: str
    url: str
    filename: str
    # Optional known checksum(s). If not provided, we'll attempt to fetch a sidecar checksum.
    sha256: str | None = None
    md5: str | None = None


OPENSLR_RESOURCES: dict[str, Resource] = {
    # LibriSpeech SLR12
    "train-clean-100": Resource("train-clean-100", "https://www.openslr.org/resources/12/train-clean-100.tar.gz", "train-clean-100.tar.gz"),
    "train-clean-360": Resource("train-clean-360", "https://www.openslr.org/resources/12/train-clean-360.tar.gz", "train-clean-360.tar.gz"),
    "train-other-500": Resource("train-other-500", "https://www.openslr.org/resources/12/train-other-500.tar.gz", "train-other-500.tar.gz"),
    "dev-clean": Resource("dev-clean", "https://www.openslr.org/resources/12/dev-clean.tar.gz", "dev-clean.tar.gz"),
    "dev-other": Resource("dev-other", "https://www.openslr.org/resources/12/dev-other.tar.gz", "dev-other.tar.gz"),
    "test-clean": Resource("test-clean", "https://www.openslr.org/resources/12/test-clean.tar.gz", "test-clean.tar.gz"),
    "test-other": Resource("test-other", "https://www.openslr.org/resources/12/test-other.tar.gz", "test-other.tar.gz"),
    # MUSAN SLR17
    "musan": Resource("musan", "https://www.openslr.org/resources/17/musan.tar.gz", "musan.tar.gz"),
    # RIRS_NOISES SLR28
    "rirs_noises": Resource("rirs_noises", "https://www.openslr.org/resources/28/rirs_noises.zip", "rirs_noises.zip"),
    # LibriSpeech LM SLR11 (optional)
    "4-gram": Resource("4-gram", "https://www.openslr.org/resources/11/4-gram.arpa.gz", "4-gram.arpa.gz"),
}


def _download(url: str, dst: Path, *, chunk_size: int = 1024 * 1024) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Support resume if server accepts Range.
    tmp = dst.with_suffix(dst.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        # If resume isn't supported, start over.
        if existing > 0 and getattr(resp, "status", 200) == 200:
            tmp.unlink(missing_ok=True)
            existing = 0

        mode = "ab" if existing > 0 else "wb"
        with tmp.open(mode) as f:
            while True:
                b = resp.read(chunk_size)
                if not b:
                    break
                f.write(b)

    tmp.replace(dst)


_MD5_LINE_RE = re.compile(r"\b([0-9a-fA-F]{32})\b\s+\*?(.+)$")


def try_fetch_sidecar_md5(url: str) -> dict[str, str] | None:
    # Common patterns seen in the wild.
    candidates = [url + ".md5", url + ".md5sum", url + ".md5.txt"]
    for c in candidates:
        try:
            with urllib.request.urlopen(c) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception:
            continue
        out: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = _MD5_LINE_RE.search(line)
            if not m:
                continue
            out[m.group(2).strip()] = m.group(1).lower()
        if out:
            return out
    return None


def verify_file(path: Path, *, sha256: str | None, md5: str | None) -> None:
    if sha256:
        got = sha256_file(path)
        if got.lower() != sha256.lower():
            raise RuntimeError(f"SHA256 mismatch for {path.name}: expected {sha256}, got {got}")
        return
    if md5:
        got = md5_file(path)
        if got.lower() != md5.lower():
            raise RuntimeError(f"MD5 mismatch for {path.name}: expected {md5}, got {got}")
        return


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir, e.g. data/raw")
    ap.add_argument("--librispeech", choices=["clean100", "ls960"], default="clean100")
    ap.add_argument("--musan", action="store_true")
    ap.add_argument("--rirs", action="store_true")
    ap.add_argument("--lm", choices=["none", "4gram"], default="none")
    ap.add_argument("--extract", action="store_true", help="extract archives after download (idempotent)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds to sleep between downloads")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    todo: list[Resource] = []
    if args.librispeech == "clean100":
        todo += [OPENSLR_RESOURCES["train-clean-100"], OPENSLR_RESOURCES["dev-clean"], OPENSLR_RESOURCES["dev-other"], OPENSLR_RESOURCES["test-clean"], OPENSLR_RESOURCES["test-other"]]
    else:
        todo += [
            OPENSLR_RESOURCES["train-clean-100"],
            OPENSLR_RESOURCES["train-clean-360"],
            OPENSLR_RESOURCES["train-other-500"],
            OPENSLR_RESOURCES["dev-clean"],
            OPENSLR_RESOURCES["dev-other"],
            OPENSLR_RESOURCES["test-clean"],
            OPENSLR_RESOURCES["test-other"],
        ]

    if args.musan:
        todo.append(OPENSLR_RESOURCES["musan"])
    if args.rirs:
        todo.append(OPENSLR_RESOURCES["rirs_noises"])
    if args.lm == "4gram":
        todo.append(OPENSLR_RESOURCES["4-gram"])

    manifest = {
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resources": [],
    }

    for r in todo:
        dst = out / r.filename
        if dst.exists():
            print(f"[skip] {dst}")
        else:
            print(f"[get] {r.url} -> {dst}")
            _download(r.url, dst)

        # Try to verify with known checksums, or by fetching a sidecar md5 list.
        sha256 = r.sha256
        md5 = r.md5
        if not sha256 and not md5:
            sidecar = try_fetch_sidecar_md5(r.url)
            if sidecar and r.filename in sidecar:
                md5 = sidecar[r.filename]

        if sha256 or md5:
            print(f"[verify] {dst.name}")
            verify_file(dst, sha256=sha256, md5=md5)
        else:
            print(f"[warn] no checksum available for {dst.name}; recording sha256 for future reproducibility")
            sha256 = sha256_file(dst)

        manifest["resources"].append(
            {
                "name": r.name,
                "url": r.url,
                "path": str(dst.resolve()),
                "sha256": sha256,
                "md5": md5,
                "bytes": dst.stat().st_size,
            }
        )

        if args.extract:
            if dst.suffixes[-2:] == [".tar", ".gz"]:
                # LibriSpeech/MUSAN tarballs extract to top-level dirs.
                print(f"[extract] {dst.name}")
                with tarfile.open(dst, mode="r:gz") as tf:
                    _safe_extract_tar(tf, out_dir=out)
            elif dst.suffix.lower() == ".zip":
                print(f"[extract] {dst.name}")
                with zipfile.ZipFile(dst) as zf:
                    _safe_extract_zip(zf, out_dir=out)
            elif dst.suffixes[-2:] == [".arpa", ".gz"] or dst.name.endswith(".arpa.gz"):
                # LM; place under out/lm
                lm_out = out / "lm"
                lm_out.mkdir(parents=True, exist_ok=True)
                out_path = lm_out / dst.name[:-3]
                if not out_path.exists():
                    print(f"[extract] {dst.name} -> {out_path.name}")
                    with gzip.open(dst, "rb") as fin, out_path.open("wb") as fout:
                        fout.write(fin.read())

        time.sleep(float(args.sleep))

    write_json(out / "download_manifest.json", manifest)


if __name__ == "__main__":
    main()
