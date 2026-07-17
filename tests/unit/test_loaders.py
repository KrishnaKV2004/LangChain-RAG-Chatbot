"""Unit tests for the format-specific document loaders."""

import json
from pathlib import Path

import pytest

from app.loaders.registry import default_registry
from app.utils.exceptions import DocumentLoadError

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# Fixtures — real files generated on the fly so parsers are exercised for real
# --------------------------------------------------------------------------- #


@pytest.fixture()
def txt_file(tmp_path: Path) -> Path:
    path = tmp_path / "notes.txt"
    path.write_text("An airway bill is a shipping document for air cargo.\n" * 5)
    return path


@pytest.fixture()
def md_file(tmp_path: Path) -> Path:
    path = tmp_path / "sop.md"
    path.write_text(
        "# Dangerous Goods SOP\n\n## Lithium Batteries\n\n"
        "Lithium batteries must be declared under UN3480.\n"
    )
    return path


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "rates.csv"
    path.write_text("origin,destination,rate\nICN,DXB,4.20\nICN,LAX,3.85\n")
    return path


@pytest.fixture()
def json_file(tmp_path: Path) -> Path:
    path = tmp_path / "faq.json"
    path.write_text(
        json.dumps(
            [
                {"question": "What is an AWB?", "answer": "Air waybill."},
                {"question": "What is FCL?", "answer": "Full container load."},
            ]
        )
    )
    return path


@pytest.fixture()
def html_file(tmp_path: Path) -> Path:
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>Customs Guide</title><script>evil()</script></head>"
        "<body><h1>Import Rules</h1><p>Declare all goods at customs.</p></body></html>"
    )
    return path


@pytest.fixture()
def docx_file(tmp_path: Path) -> Path:
    docx = pytest.importorskip("docx")
    path = tmp_path / "manual.docx"
    document = docx.Document()
    document.add_heading("Warehouse Safety", level=1)
    document.add_paragraph("Forklifts must be inspected daily before operation.")
    document.add_heading("Cold Chain", level=1)
    document.add_paragraph("Perishables are stored between 2 and 8 degrees Celsius.")
    document.save(str(path))
    return path


@pytest.fixture()
def xlsx_file(tmp_path: Path) -> Path:
    pytest.importorskip("openpyxl")
    import pandas as pd

    path = tmp_path / "tariffs.xlsx"
    pd.DataFrame({"lane": ["ICN-DXB"], "price": ["1200"]}).to_excel(
        path, sheet_name="AirFreight", index=False
    )
    return path


@pytest.fixture()
def pptx_file(tmp_path: Path) -> Path:
    pptx = pytest.importorskip("pptx")
    path = tmp_path / "training.pptx"
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "DG Handling"
    slide.placeholders[1].text = "Always segregate class 8 corrosives."
    presentation.save(str(path))
    return path


# --------------------------------------------------------------------------- #
# Base behaviour
# --------------------------------------------------------------------------- #


class TestBaseBehaviour:
    def test_base_metadata_present_on_every_document(self, txt_file: Path) -> None:
        docs = default_registry.loader_for(txt_file).load(txt_file)
        meta = docs[0].metadata
        assert meta["filename"] == "notes.txt"
        assert meta["source"].endswith("notes.txt")
        assert meta["doc_type"] == "text"
        assert "created_at" in meta and "ingested_at" in meta

    def test_missing_file_raises(self) -> None:
        with pytest.raises(DocumentLoadError, match="not found"):
            default_registry.loader_for(Path("ghost.txt")).load(Path("ghost.txt"))

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.txt"
        empty.touch()
        with pytest.raises(DocumentLoadError, match="empty"):
            default_registry.loader_for(empty).load(empty)

    def test_unsupported_extension_raises(self) -> None:
        with pytest.raises(DocumentLoadError, match="Unsupported"):
            default_registry.loader_for(Path("video.mp4"))


# --------------------------------------------------------------------------- #
# Format-specific behaviour
# --------------------------------------------------------------------------- #


class TestFormats:
    def test_csv_rows_serialized_with_column_names(self, csv_file: Path) -> None:
        docs = default_registry.loader_for(csv_file).load(csv_file)
        assert "origin: ICN" in docs[0].page_content
        assert docs[0].metadata["rows"] == "1-2"

    def test_json_array_yields_document_per_item(self, json_file: Path) -> None:
        docs = default_registry.loader_for(json_file).load(json_file)
        assert len(docs) == 2
        assert "question: What is an AWB?" in docs[0].page_content

    def test_html_strips_scripts_and_extracts_title(self, html_file: Path) -> None:
        docs = default_registry.loader_for(html_file).load(html_file)
        assert "evil()" not in docs[0].page_content
        assert "Declare all goods" in docs[0].page_content
        assert docs[0].metadata["section_title"] == "Customs Guide"

    def test_docx_splits_on_headings(self, docx_file: Path) -> None:
        docs = default_registry.loader_for(docx_file).load(docx_file)
        titles = [d.metadata.get("section_title") for d in docs]
        assert "Warehouse Safety" in titles and "Cold Chain" in titles

    def test_xlsx_sheet_name_becomes_section_title(self, xlsx_file: Path) -> None:
        docs = default_registry.loader_for(xlsx_file).load(xlsx_file)
        assert docs[0].metadata["section_title"] == "AirFreight"
        assert "lane: ICN-DXB" in docs[0].page_content

    def test_pptx_slide_number_and_title(self, pptx_file: Path) -> None:
        docs = default_registry.loader_for(pptx_file).load(pptx_file)
        assert docs[0].metadata["page"] == 1
        assert docs[0].metadata["section_title"] == "DG Handling"
        assert "class 8" in docs[0].page_content

    @pytest.mark.skipif(
        not (REPO_ROOT / "documents" / "Airports.pdf").exists(),
        reason="sample PDF not present",
    )
    def test_pdf_page_numbers(self) -> None:
        pdf = REPO_ROOT / "documents" / "Airports.pdf"
        docs = default_registry.loader_for(pdf).load(pdf)
        assert docs[0].metadata["page"] == 1
        assert docs[0].metadata["doc_type"] == "pdf"
        assert all("total_pages" in d.metadata for d in docs)
