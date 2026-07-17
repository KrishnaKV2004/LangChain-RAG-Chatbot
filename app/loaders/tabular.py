"""Loaders for tabular formats: CSV and Excel.

Rows are serialized as human-readable ``column: value`` lines and grouped
into row batches so that (a) each Document stays within embedding-friendly
size, and (b) the ``rows`` metadata range lets citations point at the exact
slice of the spreadsheet ("rates.csv rows 26–50").
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.loaders.base import BaseDocumentLoader

#: Rows serialized into a single Document before the chunker takes over.
_ROWS_PER_DOCUMENT = 25


def _serialize_rows(df: "object", start: int, end: int) -> str:
    """Render dataframe rows [start:end) as 'col: value; col: value' lines."""
    lines: List[str] = []
    for _, row in df.iloc[start:end].iterrows():  # type: ignore[attr-defined]
        pairs = [
            f"{column}: {value}"
            for column, value in row.items()
            if value is not None and str(value).strip() not in ("", "nan", "NaN")
        ]
        if pairs:
            lines.append("; ".join(pairs))
    return "\n".join(lines)


def _dataframe_to_documents(df: "object", extra_metadata: dict) -> List[Document]:
    """Split a dataframe into row-batched Documents with range metadata."""
    documents: List[Document] = []
    total = len(df)  # type: ignore[arg-type]
    for start in range(0, total, _ROWS_PER_DOCUMENT):
        end = min(start + _ROWS_PER_DOCUMENT, total)
        text = _serialize_rows(df, start, end)
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                # 1-based inclusive range, matching how humans read sheets.
                metadata={"rows": f"{start + 1}-{end}", **extra_metadata},
            )
        )
    return documents


class CSVLoader(BaseDocumentLoader):
    """CSV loader with delimiter sniffing via pandas."""

    suffixes = (".csv",)
    doc_type = "csv"

    def _load(self, path: Path) -> List[Document]:
        import pandas as pd

        # sep=None + python engine autodetects the delimiter (, ; \t |).
        df = pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)
        return _dataframe_to_documents(df, extra_metadata={})


class ExcelLoader(BaseDocumentLoader):
    """Excel loader: every sheet is processed, sheet name → section_title."""

    suffixes = (".xlsx", ".xls", ".xlsm")
    doc_type = "excel"

    def _load(self, path: Path) -> List[Document]:
        import pandas as pd

        sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)
        documents: List[Document] = []
        for sheet_name, df in sheets.items():
            documents.extend(
                _dataframe_to_documents(df, extra_metadata={"section_title": str(sheet_name)})
            )
        return documents
