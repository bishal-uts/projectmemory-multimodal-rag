"""Bounded remote ZIP retrieval tests with a local in-memory HTTP stand-in."""

import hashlib
import io
import re
import stat
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from projectmemory.remote_zip import RemoteZipError, fetch_selected_zip


URL = "https://example.test/original-images.zip"


def make_zip(entries, *, streaming=False):
    class NonSeekingBuffer(io.BytesIO):
        def seek(self, *args):
            raise io.UnsupportedOperation("not seekable")

    output = NonSeekingBuffer() if streaming else io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return output.getvalue()


class FakeResponse(io.BytesIO):
    def __init__(self, payload, status, headers, server):
        super().__init__(payload)
        self.status, self.headers, self.server = status, headers, server

    def read(self, size=-1):
        payload = super().read(size)
        self.server.bytes_read += len(payload)
        return payload


class RangeServer:
    def __init__(self, archive, *, range_supported=True, wrong_range=False, changed_etag=False):
        self.archive = archive
        self.range_supported = range_supported
        self.wrong_range, self.changed_etag = wrong_range, changed_etag
        self.calls = []
        self.bytes_read = 0

    def __call__(self, request, timeout=None):
        self.calls.append(request)
        etag = '"original-archive"'
        if request.get_method() == "HEAD":
            return FakeResponse(b"", 200, {"Content-Length": str(len(self.archive)), "ETag": etag}, self)
        if not self.range_supported:
            return FakeResponse(self.archive, 200, {"Content-Length": str(len(self.archive))}, self)
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        if not match:
            raise AssertionError("Missing a bounded byte range")
        start, end = map(int, match.groups())
        payload = self.archive[start:end + 1]
        headers = {"Content-Range": f"bytes {start}-{end}/{len(self.archive)}",
                   "Content-Length": str(len(payload)),
                   "ETag": '"changed"' if self.changed_etag else etag}
        if self.wrong_range:
            headers["Content-Range"] = f"bytes 0-0/{len(self.archive)}"
        return FakeResponse(payload, 206, headers, self)


class RemoteZipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_selective_original_members_with_prefix_and_streaming_headers(self):
        original = {"a.JPG": b"original first image", "b.png": b"original second image"}
        archive = make_zip([(f"final_dataset_images/{name}", payload) for name, payload in original.items()]
                           + [("answers.json", b"never extract gold answer material")], streaming=True)
        server = RangeServer(archive)
        outputs = {name: self.directory / name for name in original}
        with patch("projectmemory.remote_zip.urlopen", server):
            report = fetch_selected_zip(URL, outputs, workers=2)
        self.assertEqual(report["downloaded"], 2)
        self.assertEqual(report["skipped"], 0)
        self.assertEqual(report["bytes_downloaded"], server.bytes_read)
        self.assertEqual(report["bytes_written"], sum(map(len, original.values())))
        self.assertEqual(report["archive_bytes"], len(archive))
        for name, payload in original.items():
            self.assertEqual(outputs[name].read_bytes(), payload)
        self.assertFalse((self.directory / "answers.json").exists())
        self.assertEqual(len(list(self.directory.iterdir())), 2)
        self.assertEqual(report["files"][0]["sha256"], hashlib.sha256(original["a.JPG"]).hexdigest())
        self.assertTrue(all(request.get_header("If-match") == '"original-archive"'
                            for request in server.calls if request.get_method() == "GET"))

    def test_existing_files_are_skipped_only_after_crc_validation(self):
        archive = make_zip([("prefix/a.jpg", b"valid bytes")])
        destination = self.directory / "a.jpg"
        destination.write_bytes(b"wrong bytes")  # Same size; checksum matters.
        server = RangeServer(archive)
        with patch("projectmemory.remote_zip.urlopen", server):
            repaired = fetch_selected_zip(URL, {"a.jpg": destination})
            skipped = fetch_selected_zip(URL, {"a.jpg": destination})
        self.assertEqual(repaired["downloaded"], 1)
        self.assertEqual(destination.read_bytes(), b"valid bytes")
        self.assertEqual(skipped["skipped"], 1)
        self.assertEqual(skipped["downloaded"], 0)
        self.assertEqual(skipped["bytes_written"], 0)

    def test_server_ignoring_range_is_rejected_before_body_read(self):
        server = RangeServer(make_zip([("a.jpg", b"image")]), range_supported=False)
        with patch("projectmemory.remote_zip.urlopen", server):
            with self.assertRaisesRegex(RemoteZipError, "whole-archive download refused"):
                fetch_selected_zip(URL, {"a.jpg": self.directory / "a.jpg"})
        self.assertEqual(server.bytes_read, 0)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_budget_covers_directory_and_rejects_oversized_uncompressed_content(self):
        server = RangeServer(make_zip([("a.jpg", b"image")]))
        with patch("projectmemory.remote_zip.urlopen", server):
            with self.assertRaisesRegex(RemoteZipError, "budget"):
                fetch_selected_zip(URL, {"a.jpg": self.directory / "a.jpg"}, max_bytes=10)
        self.assertEqual(server.bytes_read, 0)
        bomb = RangeServer(make_zip([("a.jpg", b"0" * 100_000)]))
        with patch("projectmemory.remote_zip.urlopen", bomb):
            with self.assertRaisesRegex(RemoteZipError, "per-file byte limit"):
                fetch_selected_zip(URL, {"a.jpg": self.directory / "a.jpg"}, max_bytes=5_000)
        self.assertLess(bomb.bytes_read, 5_000)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_archive_traversal_and_symlinks_are_rejected(self):
        link = zipfile.ZipInfo("link.jpg")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        entries = [("../escape.jpg", b"bad"), ("C:/escape.jpg", b"bad"), (link, b"target")]
        cases = [make_zip([("a.jpg", b"image"), entry]) for entry in entries]
        # zipfile normalizes Windows backslashes on creation. Mutate both stored
        # filenames afterwards to exercise a hostile archive from another source.
        cases.append(make_zip([("a.jpg", b"image"), ("dir/escape.jpg", b"bad")]).replace(
            b"dir/escape.jpg", b"dir\\escape.jpg"))
        for archive in cases:
            with self.subTest(case=cases.index(archive)):
                server = RangeServer(archive)
                with patch("projectmemory.remote_zip.urlopen", server):
                    with self.assertRaises(RemoteZipError):
                        fetch_selected_zip(URL, {"a.jpg": self.directory / "a.jpg"})
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_ambiguous_basename_fails_but_exact_archive_path_works(self):
        server = RangeServer(make_zip([("one/a.jpg", b"one"), ("two/a.jpg", b"two")]))
        with patch("projectmemory.remote_zip.urlopen", server):
            with self.assertRaisesRegex(RemoteZipError, "ambiguous"):
                fetch_selected_zip(URL, {"a.jpg": self.directory / "a.jpg"})
            report = fetch_selected_zip(URL, {"two/a.jpg": self.directory / "a.jpg"})
        self.assertEqual(report["downloaded"], 1)
        self.assertEqual((self.directory / "a.jpg").read_bytes(), b"two")

    def test_crc_failure_never_writes_corrupt_image(self):
        archive = bytearray(make_zip([("a.jpg", b"original image")]))
        position = archive.index(b"PK\x01\x02")
        old_crc, = struct.unpack_from("<L", archive, position + 16)
        struct.pack_into("<L", archive, position + 16, old_crc ^ 1)
        server = RangeServer(bytes(archive))
        with patch("projectmemory.remote_zip.urlopen", server):
            with self.assertRaisesRegex(RemoteZipError, "CRC"):
                fetch_selected_zip(URL, {"a.jpg": self.directory / "a.jpg"})
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_changed_archive_or_wrong_range_fails(self):
        archive = make_zip([("a.jpg", b"image")])
        for options in ({"changed_etag": True}, {"wrong_range": True}):
            with self.subTest(options=options):
                with patch("projectmemory.remote_zip.urlopen", RangeServer(archive, **options)):
                    with self.assertRaises(RemoteZipError):
                        fetch_selected_zip(URL, {"a.jpg": self.directory / "a.jpg"})
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_missing_member_and_duplicate_destination_are_rejected(self):
        server = RangeServer(make_zip([("a.jpg", b"image")]))
        with patch("projectmemory.remote_zip.urlopen", server):
            with self.assertRaisesRegex(RemoteZipError, "missing"):
                fetch_selected_zip(URL, {"missing.jpg": self.directory / "a.jpg"})
            with self.assertRaisesRegex(RemoteZipError, "same output"):
                fetch_selected_zip(URL, {"a.jpg": self.directory / "a.jpg",
                                         "b.jpg": self.directory / "a.jpg"})

    def test_invalid_request_paths_and_empty_selection_never_use_network(self):
        with patch("projectmemory.remote_zip.urlopen") as network:
            self.assertEqual(fetch_selected_zip(URL, {})["bytes_downloaded"], 0)
            with self.assertRaises(RemoteZipError):
                fetch_selected_zip(URL, {"../a.jpg": self.directory / "a.jpg"})
            with self.assertRaises(ValueError):
                fetch_selected_zip(URL, {"a.jpg": Path("relative.jpg")})
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
