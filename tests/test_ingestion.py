from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import pytest

from projectmemory.ingestion import ingest_file, ingest_paths
from projectmemory.io import read_jsonl


def test_text_overlap_and_reingest_preserve_evidence(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text(" ".join(f"word{i}" for i in range(500)), encoding="utf-8")
    output = tmp_path / "prepared"
    first = ingest_paths([source], output)
    second = ingest_paths([source], output)
    assert first["corpus_records"] == second["corpus_records"] == 3
    rows = read_jsonl(output / "corpus.jsonl")
    assert "word0" in rows[0]["text"] and "word499" in rows[-1]["text"]
    assert rows[1]["metadata"]["start_word"] == 180
    assert read_jsonl(output / "queries.jsonl") == []


def test_html_excludes_scripts_and_keeps_content(tmp_path):
    source = tmp_path / "page.html"
    source.write_text("<h1>Meeting notes</h1><script>secret_call()</script><p>Use blue.</p>")
    result = ingest_file(source, tmp_path / "assets")
    assert "Use blue." in result[0]["text"]
    assert "secret_call" not in result[0]["text"]


def test_pdf_retains_page_pixels_and_text(tmp_path):
    import pymupdf
    source = tmp_path / "fixture.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Synthetic ingestion test: blue remote control.")
        document.save(source)
    rows = ingest_file(source, tmp_path / "assets")
    assert len(rows) == 1 and rows[0]["modality"] == "image"
    assert "blue remote" in rows[0]["text"]
    assert rows[0]["metadata"]["page"] == 1
    assert Path(rows[0]["path"]).is_file()


def test_office_parser_extracts_text_without_extracting_archive_paths(tmp_path):
    source = tmp_path / "fixture.docx"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="urn:test"><w:p><w:t>Meeting evidence</w:t></w:p></w:document>')
        archive.writestr("../../escape.txt", "must not be extracted")
    rows = ingest_file(source, tmp_path / "assets")
    assert rows[0]["text"] == "Meeting evidence"
    assert not (tmp_path / "escape.txt").exists()


def _write_pptx(source, *, relationships=None, omitted=(), strict=False):
    """Small synthetic package whose display order differs from ZIP filenames."""
    presentation_ns = ("http://purl.oclc.org/ooxml/presentationml/main" if strict else
                       "http://schemas.openxmlformats.org/presentationml/2006/main")
    relationship_ns = ("http://purl.oclc.org/ooxml/officeDocument/relationships" if strict else
                       "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
    package_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    presentation = ET.Element(f"{{{presentation_ns}}}presentation")
    slide_list = ET.SubElement(presentation, f"{{{presentation_ns}}}sldIdLst")
    for index, identity in enumerate(("rId2", "rId1"), 256):
        ET.SubElement(slide_list, f"{{{presentation_ns}}}sldId",
                      {"id": str(index), f"{{{relationship_ns}}}id": identity})
    relationship_root = ET.Element(f"{{{package_ns}}}Relationships")
    if relationships is None:
        relationships = [{"Id": f"rId{number}", "Type": relationship_ns + "/slide",
                          "Target": f"slides/slide{number}.xml"} for number in (1, 2)]
    for attributes in relationships:
        ET.SubElement(relationship_root, f"{{{package_ns}}}Relationship", attributes)
    parts = {"ppt/presentation.xml": ET.tostring(presentation),
             "ppt/_rels/presentation.xml.rels": ET.tostring(relationship_root)}
    for number, text in ((1, "Second displayed slide"), (2, "First displayed slide"),
                         (3, "Orphan slide must not be imported")):
        parts[f"ppt/slides/slide{number}.xml"] = (
            '<p:sld xmlns:p="urn:fixture:p" xmlns:a="urn:fixture:a">'
            f'<a:t>{text}</a:t></p:sld>')
    with zipfile.ZipFile(source, "w") as archive:
        for name, content in parts.items():
            if name not in omitted:
                archive.writestr(name, content)


@pytest.mark.parametrize("strict", [False, True])
def test_pptx_citations_follow_presentation_order_and_skip_orphan_slides(tmp_path, strict):
    source = tmp_path / "reordered.pptx"
    _write_pptx(source, strict=strict)
    rows = ingest_file(source, tmp_path / "assets")
    assert [(row["metadata"]["slide"], row["text"]) for row in rows] == [
        (1, "First displayed slide"), (2, "Second displayed slide")]
    assert rows[0]["title"] == "reordered.pptx, slide 1"
    assert rows[0]["metadata"]["archive_member"] == "ppt/slides/slide2.xml"
    assert ":slide:1:" in rows[0]["id"]


@pytest.mark.parametrize("change, message", [
    ({"Id": "missing"}, "missing or invalid relationship"),
    ({"Target": "slides/missing.xml"}, "missing XML part"),
    ({"Target": "../../escape.xml"}, "escapes the package"),
    ({"TargetMode": "External", "Target": "https://example.com/slide.xml"}, "must be internal"),
    ({"Target": "https://example.com/slide.xml"}, "invalid internal target"),
    ({"Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"}, "not a slide"),
    ({"Id": "rId2"}, "duplicate relationship IDs"),
])
def test_pptx_invalid_slide_relationships_fail_without_inventing_locations(tmp_path, change, message):
    source = tmp_path / "invalid.pptx"
    kind = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
    relationships = [{"Id": f"rId{number}", "Type": kind, "Target": f"slides/slide{number}.xml"}
                     for number in (1, 2)]
    relationships[0].update(change)
    _write_pptx(source, relationships=relationships)
    with pytest.raises(ValueError, match=message):
        ingest_file(source, tmp_path / "assets")
    assert not (tmp_path / "escape.xml").exists()


@pytest.mark.parametrize("omitted", ["ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"])
def test_pptx_requires_presentation_order_parts(tmp_path, omitted):
    source = tmp_path / "missing-order.pptx"
    _write_pptx(source, omitted=(omitted,))
    with pytest.raises(ValueError, match="valid presentation XML and slide relationships"):
        ingest_file(source, tmp_path / "assets")
