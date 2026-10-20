"""Archive fixtures are synthetic bytes, never real benchmark media."""

import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from projectmemory.io import read_jsonl, write_json, write_jsonl
from projectmemory.media import IMAGE_ARCHIVE_URL, VIDEO_URL, extract_selected, fetch_media


def record(root: Path, member: str, identity: str = "mmqa:image:fixture") -> dict:
    return {"id": identity, "archive_member": member, "path": str((root / member).resolve())}


class SafeExtractionTests(unittest.TestCase):
    def test_zip_extracts_only_manifest_members_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            base = Path(directory)
            media = base / "media"
            archive = base / "fixtures.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("keep.jpg", b"selected synthetic bytes")
                output.writestr("ignore.jpg", b"unselected synthetic bytes")
            result = extract_selected(archive, [record(media, "keep.jpg")], media)
            self.assertEqual(result["extracted_ids"], ["mmqa:image:fixture"])
            self.assertEqual((media / "keep.jpg").read_bytes(), b"selected synthetic bytes")
            self.assertFalse((media / "ignore.jpg").exists())
            (media / "keep.jpg").write_bytes(b"preserve my file")
            second = extract_selected(archive, [record(media, "keep.jpg")], media)
            self.assertEqual(second["already_available_ids"], ["mmqa:image:fixture"])
            self.assertEqual((media / "keep.jpg").read_bytes(), b"preserve my file")

    def test_tar_extracts_exact_nested_member_and_reports_missing(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            base = Path(directory)
            media = base / "media"
            archive = base / "fixtures.tar"
            member = "Videos/clip/clip_video.mp4"
            with tarfile.open(archive, "w") as output:
                info = tarfile.TarInfo("./" + member)
                info.size = 4
                output.addfile(info, io.BytesIO(b"fake"))
            required = [record(media, member), record(media, "absent.wav", "absent")]
            result = extract_selected(archive, required, media)
            self.assertEqual((media / member).read_bytes(), b"fake")
            self.assertEqual(result["missing_member_ids"], ["absent"])

    def test_zip_traversal_is_rejected_before_any_extraction(self):
        for bad_path in ("../escape.jpg", "/absolute.jpg", "C:/escape.jpg", "dir\\escape.jpg"):
            with self.subTest(path=bad_path), tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                base = Path(directory)
                media = base / "media"
                archive = base / "bad.zip"
                with zipfile.ZipFile(archive, "w") as output:
                    output.writestr("safe.jpg", b"safe")
                    output.writestr(bad_path, b"unsafe")
                if "\\" in bad_path:
                    archive.write_bytes(archive.read_bytes().replace(bad_path.replace("\\", "/").encode(), bad_path.encode()))
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    extract_selected(archive, [record(media, "safe.jpg")], media)
                self.assertFalse((media / "safe.jpg").exists())

    def test_tar_traversal_and_links_are_rejected(self):
        for bad_type in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, "traversal"):
            with self.subTest(type=bad_type), tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                base = Path(directory)
                archive = base / "bad.tar"
                with tarfile.open(archive, "w") as output:
                    safe = tarfile.TarInfo("safe.jpg")
                    safe.size = 4
                    output.addfile(safe, io.BytesIO(b"safe"))
                    bad = tarfile.TarInfo("../escape" if bad_type == "traversal" else "link")
                    if bad_type != "traversal":
                        bad.type = bad_type
                        bad.linkname = "outside"
                    output.addfile(bad)
                with self.assertRaises(ValueError):
                    extract_selected(archive, [record(base / "media", "safe.jpg")], base / "media")
                self.assertFalse((base / "media/safe.jpg").exists())

    def test_zip_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            base = Path(directory)
            archive = base / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                info = zipfile.ZipInfo("link")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                output.writestr(info, "../outside")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                extract_selected(archive, [record(base / "media", "link")], base / "media")

    def test_manifest_cannot_write_outside_media_root(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            base = Path(directory)
            archive = base / "test.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("safe.jpg", b"fake")
            escaped = {"id": "escape", "archive_member": "safe.jpg", "path": str(base / "outside.jpg")}
            with self.assertRaisesRegex(ValueError, "leaves"):
                extract_selected(archive, [escaped], base / "media")
            self.assertFalse((base / "outside.jpg").exists())

    def test_declared_expansion_budget_fails_before_writing(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            base = Path(directory)
            archive = base / "large.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                output.writestr("large.jpg", b"x" * 10_000)
            with self.assertRaisesRegex(ValueError, "uncompressed"):
                extract_selected(archive, [record(base / "media", "large.jpg")], base / "media", max_bytes=100)
            self.assertFalse((base / "media/large.jpg").exists())


class FetchMediaTests(unittest.TestCase):
    def test_selective_mmqa_fetch_limits_files_and_refreshes_state(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            data_root = Path(directory) / "data"
            prepared = data_root / "prepared/mmqa"
            media = data_root / "media/mmqa/images"
            required = [record(media, "first.jpg", "mmqa:image:first"),
                        record(media, "second.jpg", "mmqa:image:second")]
            write_json(prepared / "download_manifest.json", {
                "dataset": "multimodalqa", "required_media": required,
                "archives": [{"url": IMAGE_ARCHIVE_URL, "destination": str(media.resolve())}],
            })
            write_json(prepared / "manifest.json", {"dataset": "multimodalqa", "native_multimodal_ready": False})
            write_jsonl(prepared / "corpus.jsonl", [dict(item, modality="image", metadata={}) for item in required])
            calls = []

            def mock_ranges(url, members, max_bytes):
                calls.append((url, max_bytes))
                for path in members.values():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"synthetic range-fetched image")
                return {"downloaded": len(members), "bytes_downloaded": 200,
                        "bytes_written": 28 * len(members), "archive_bytes": 2_000_000_000}

            with patch("projectmemory.remote_zip.fetch_selected_zip", side_effect=mock_ranges):
                first = fetch_media(prepared, max_bytes=1_000_000, limit=1)
                self.assertEqual(first["extracted_media"], 1)
                self.assertEqual(first["media_missing"], 1)
                self.assertFalse(first["complete"])
                self.assertEqual(calls[0], (IMAGE_ARCHIVE_URL, 1_000_000))
                self.assertFalse((media / "second.jpg").exists())
                second = fetch_media(prepared, max_bytes=1_000_000)
                self.assertTrue(second["complete"])
                self.assertEqual(second["archives"][0]["strategy"], "selective_zip_ranges")
                third = fetch_media(prepared, max_bytes=1_000_000)
                self.assertEqual(third["downloaded_bytes"], 0)
                self.assertEqual(len(calls), 2)
            self.assertTrue(all(row["metadata"]["media_available"] for row in read_jsonl(prepared / "corpus.jsonl")))
            manifest = json.loads((prepared / "manifest.json").read_text())
            self.assertTrue(manifest["native_multimodal_ready"])
            self.assertEqual(manifest["missing_image_files"], 0)

    def test_daily_archive_is_bounded_cached_and_selectively_extracted(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            data_root = Path(directory) / "data"
            prepared = data_root / "prepared/daily-omni"
            media = data_root / "media/daily_omni"
            required = [record(media, "Videos/first/first_video.mp4", "daily_omni:video:first"),
                        record(media, "Videos/second/second_video.mp4", "daily_omni:video:second")]
            write_json(prepared / "download_manifest.json", {
                "dataset": "daily_omni", "required_media": required,
                "archives": [{"url": VIDEO_URL, "destination": str(media.resolve())}],
            })
            budgets = []

            def mock_download(url, path, max_bytes):
                budgets.append(max_bytes)
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    with tarfile.open(path, "w") as output:
                        for item in required:
                            info = tarfile.TarInfo(item["archive_member"])
                            info.size = 4
                            output.addfile(info, io.BytesIO(b"fake"))
                return path

            with patch("projectmemory.media.download", side_effect=mock_download):
                first = fetch_media(prepared, max_bytes=1_000_000, limit=1)
                self.assertEqual(first["extracted_media"], 1)
                self.assertGreater(first["downloaded_bytes"], 0)
                second = fetch_media(prepared, max_bytes=1_000_000)
                self.assertTrue(second["complete"])
                self.assertTrue(second["archives"][0]["cache_hit"])
                self.assertEqual(second["downloaded_bytes"], 0)
            self.assertEqual(budgets, [1_000_000, 1_000_000])

    def test_foreign_archive_url_rejected_before_network(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            base = Path(directory) / "data"
            prepared = base / "prepared/mmqa"
            media = base / "media"
            write_json(prepared / "download_manifest.json", {
                "dataset": "multimodalqa", "required_media": [record(media, "first.jpg")],
                "archives": [{"url": "https://example.invalid/replace.zip", "destination": str(media.resolve())}],
            })
            with patch("projectmemory.media.download", side_effect=AssertionError("Unexpected network")):
                with self.assertRaisesRegex(ValueError, "official"):
                    fetch_media(prepared)


if __name__ == "__main__":
    unittest.main()
