"""Prepare local files as source evidence with page, word or time citations."""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
from pathlib import Path
import posixpath
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import zipfile

from .io import read_jsonl, write_json, write_jsonl

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".html", ".htm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
AV_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | AV_EXTENSIONS | {".pdf", ".docx", ".pptx"}


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            checksum.update(block)
    return checksum.hexdigest()


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _text_chunks(text: str, base: dict, *, words_per_chunk=220, overlap=40):
    words = text.split()
    for start in range(0, len(words), words_per_chunk - overlap):
        end = min(start + words_per_chunk, len(words))
        yield {**base, "id": f"{base['id']}:words:{start:06d}-{end:06d}",
               "modality": "text", "text": " ".join(words[start:end]),
               "metadata": {**base["metadata"], "start_word": start, "end_word_exclusive": end}}
        if end == len(words):
            break


def _image_record(data: bytes | Path, base: dict, output: Path) -> dict:
    from PIL import Image, ImageOps
    import io
    with Image.open(data if isinstance(data, Path) else io.BytesIO(data)) as original:
        if original.width * original.height > 30_000_000:
            raise ValueError("Image exceeds the 30-megapixel ingestion limit")
        image = ImageOps.exif_transpose(original).convert("RGB")
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, format="PNG")
    return {**base, "modality": "image", "text": base.get("text", ""),
            "path": str(output.resolve())}


def _xml(archive: zipfile.ZipFile, name: str):
    info = archive.getinfo(name)
    if info.file_size > 20_000_000:
        raise ValueError("Office XML part exceeds the 20 MB parsing limit")
    data = archive.read(name)
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("Office XML entity declarations are unsupported")
    return ET.fromstring(data)


def _pptx_slides(archive: zipfile.ZipFile) -> list[str]:
    """Resolve slide parts in presentation order without extracting ZIP paths."""
    try:
        presentation = _xml(archive, "ppt/presentation.xml")
        relationships = _xml(archive, "ppt/_rels/presentation.xml.rels")
    except (KeyError, ET.ParseError) as error:
        raise ValueError("PPTX requires valid presentation XML and slide relationships") from error
    presentation_namespaces = {
        "http://schemas.openxmlformats.org/presentationml/2006/main",
        "http://purl.oclc.org/ooxml/presentationml/main",
    }
    relationship_namespaces = {
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "http://purl.oclc.org/ooxml/officeDocument/relationships",
    }
    namespace = next((ns for ns in presentation_namespaces
                      if presentation.tag == f"{{{ns}}}presentation"), None)
    if namespace is None:
        raise ValueError("PPTX presentation has an unsupported XML namespace")
    package_namespace = "http://schemas.openxmlformats.org/package/2006/relationships"
    if relationships.tag != f"{{{package_namespace}}}Relationships":
        raise ValueError("PPTX slide relationships have an invalid XML root")
    by_id = {}
    for relationship in relationships.findall(f"{{{package_namespace}}}Relationship"):
        identity = relationship.get("Id")
        if not identity or identity in by_id:
            raise ValueError("PPTX contains missing or duplicate relationship IDs")
        by_id[identity] = relationship
    slide_list = presentation.find(f"{{{namespace}}}sldIdLst")
    if slide_list is None:
        return []
    members = set(archive.namelist())
    slides = []
    for slide in slide_list.findall(f"{{{namespace}}}sldId"):
        identities = [slide.get(f"{{{ns}}}id") for ns in relationship_namespaces
                      if slide.get(f"{{{ns}}}id")]
        if len(identities) != 1 or identities[0] not in by_id:
            raise ValueError("PPTX slide order references a missing or invalid relationship")
        relationship = by_id[identities[0]]
        if relationship.get("Type") not in {ns + "/slide" for ns in relationship_namespaces}:
            raise ValueError("PPTX slide order references a relationship that is not a slide")
        if relationship.get("TargetMode", "Internal") != "Internal":
            raise ValueError("PPTX slide relationships must be internal package parts")
        target = unquote(relationship.get("Target", ""))
        parsed = urlsplit(target)
        if (not target or parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
                or "\\" in target or any(ord(char) < 32 for char in target)):
            raise ValueError("PPTX slide relationship has an invalid internal target")
        member = posixpath.normpath(target.lstrip("/") if target.startswith("/")
                                   else posixpath.join("ppt", target))
        if member == ".." or member.startswith("../"):
            raise ValueError("PPTX slide relationship escapes the package")
        if member not in members or not member.lower().endswith(".xml"):
            raise ValueError("PPTX slide relationship references a missing XML part")
        slides.append(member)
    return slides


def ingest_file(path: Path, assets: Path, *, chunk_seconds=20, max_duration_seconds=None) -> list[dict]:
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported or missing source file: {path.name}")
    if path.stat().st_size > 500_000_000:
        raise ValueError("Split files larger than 500 MB before ingestion")
    digest = _digest(path)
    source_id = f"file:{digest[:24]}"
    asset_dir = Path(assets).resolve() / digest
    base = {"id": source_id, "source_id": source_id, "title": path.name,
            "metadata": {"dataset": "local_files", "source_file": str(path), "sha256": digest}}
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        text = path.read_text(encoding="utf-8-sig")
        if suffix in {".html", ".htm"}:
            parser = _HTMLText()
            parser.feed(text)
            text = "\n".join(parser.parts)
        return list(_text_chunks(text, base))
    if suffix in IMAGE_EXTENSIONS:
        return [_image_record(path, base, asset_dir / "image.png")]
    if suffix == ".pdf":
        import pymupdf
        records = []
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise ValueError("Password-protected PDFs must be unlocked before ingestion")
            if len(document) > 300:
                raise ValueError("Split PDFs longer than 300 pages before ingestion")
            asset_dir.mkdir(parents=True, exist_ok=True)
            for index, page in enumerate(document):
                text = page.get_text()
                image_path = asset_dir / f"page-{index + 1:04d}.png"
                scale = min(1.5, 1800 / max(page.rect.width, page.rect.height))
                page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).save(image_path)
                records.append({**base, "id": f"{source_id}:page:{index + 1}",
                                "title": f"{path.name}, page {index + 1}", "modality": "image",
                                "path": str(image_path), "text": text[:12000],
                                "metadata": {**base["metadata"], "page": index + 1,
                                             "representation": "rendered_pdf_page",
                                             "text_is_excerpt": len(text) > 12000}})
        return records
    if suffix in {".docx", ".pptx"}:
        records = []
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 200_000_000:
                raise ValueError("Office file expands beyond the 200 MB parsing limit")
            if suffix == ".docx":
                root = _xml(archive, "word/document.xml")
                text = "\n".join(" ".join(node.itertext()) for node in root.iter()
                                 if node.tag.endswith("}t"))
                records.extend(_text_chunks(text, base))
            else:
                slides = _pptx_slides(archive)
                for slide_number, name in enumerate(slides, 1):
                    root = _xml(archive, name)
                    text = "\n".join("".join(node.itertext()) for node in root.iter() if node.tag.endswith("}t"))
                    slide = {**base, "id": f"{source_id}:slide:{slide_number}",
                             "title": f"{path.name}, slide {slide_number}",
                             "metadata": {**base["metadata"], "slide": slide_number,
                                          "archive_member": name}}
                    records.extend(_text_chunks(text, slide))
            prefix = "word/media/" if suffix == ".docx" else "ppt/media/"
            for number, name in enumerate(sorted(archive.namelist())):
                if name.startswith(prefix) and Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                    if archive.getinfo(name).file_size > 30_000_000:
                        raise ValueError("Embedded Office image exceeds 30 MB")
                    image_base = {**base, "id": f"{source_id}:embedded:{number}",
                                  "title": f"{path.name}, embedded image {number}",
                                  "metadata": {**base["metadata"], "archive_member": name,
                                               "representation": "embedded_image_unmapped_to_page"}}
                    records.append(_image_record(archive.read(name), image_base, asset_dir / f"embedded-{number}.png"))
        return records
    from .av import segment_media
    return segment_media(path, asset_dir, chunk_seconds=chunk_seconds,
                         max_duration_seconds=max_duration_seconds)


def ingest_paths(paths: list[Path], output_dir: Path, *, chunk_seconds=20,
                 max_duration_seconds=None) -> dict:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    records = {row["id"]: row for row in read_jsonl(corpus_path)} if corpus_path.exists() else {}
    added = []
    for path in paths:
        rows = ingest_file(Path(path), output_dir / "assets", chunk_seconds=chunk_seconds,
                           max_duration_seconds=max_duration_seconds)
        if not rows:
            raise ValueError(f"No source evidence was extracted from {Path(path).name}")
        replaced_sources = {row["source_id"] for row in rows}
        records = {key: row for key, row in records.items() if row.get("source_id") not in replaced_sources}
        records.update((row["id"], row) for row in rows)
        added.append({"file": str(Path(path).resolve()), "records": len(rows)})
    write_jsonl(corpus_path, sorted(records.values(), key=lambda row: row["id"]))
    query_path = output_dir / "queries.jsonl"
    if not query_path.exists():
        write_jsonl(query_path, [])
    report = {"dataset": "local_files", "corpus_records": len(records), "last_ingested": added,
              "protocol": "user_collection_no_gold_labels", "chunk_seconds": chunk_seconds,
              "notes": ["PDF pages retain rendered pixels and extracted text; long text excerpts are flagged.",
                        "Office documents retain text and embedded images; page layout is not rendered.",
                        "Audio/video clips are physically segmented; visual clips have no audio track."]}
    write_json(output_dir / "manifest.json", report)
    return report
