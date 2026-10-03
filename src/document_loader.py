from __future__ import annotations

import csv
from datetime import date, datetime, time
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO
import pandas as pd

from docx import Document as DocxDocument
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pptx import Presentation


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
    ".pptx",
    ".xlsx",
}


def _get_filename(file_source: str | Path | BinaryIO) -> str:
    """Return the original filename from a path or uploaded file."""

    if isinstance(file_source, (str, Path)):
        return Path(file_source).name

    filename = getattr(file_source, "name", "")

    if not filename:
        raise ValueError("Uploaded file must have a filename.")

    return Path(filename).name


def _get_extension(file_source: str | Path | BinaryIO) -> str:
    """Return the normalized file extension."""

    filename = _get_filename(file_source)
    extension = Path(filename).suffix.lower()

    if not extension:
        raise ValueError(
            f"File '{filename}' has no extension."
        )

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {supported}"
        )

    return extension


def _read_uploaded_bytes(file: BinaryIO) -> bytes:
    """Read bytes from Streamlit UploadedFile or a generic binary stream."""

    if hasattr(file, "getvalue"):
        file_bytes = file.getvalue()
    else:
        file_bytes = file.read()

    if not isinstance(file_bytes, bytes):
        raise ValueError("Uploaded file must provide binary content.")

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    return file_bytes


def _build_metadata(
    filename: str,
    file_type: str,
    **extra_metadata,
) -> dict:
    """Create normalized metadata shared by all document formats."""

    metadata = {
        "source": filename,
        "document_name": filename,
        "file_type": file_type,
    }

    metadata.update(
        {
            key: value
            for key, value in extra_metadata.items()
            if value is not None
        }
    )

    return metadata


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _load_pdf_from_path(file_path: str | Path) -> list[Document]:
    """Load a PDF from an existing filesystem path."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            "Expected a PDF file."
        )

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    if not documents:
        raise ValueError(
            f"No readable content found in PDF: {path}"
        )

    filename = path.name

    for document in documents:
        document.metadata.update(
            _build_metadata(
                filename=filename,
                file_type="pdf",
            )
        )

    return documents


def _load_pdf_from_upload(file: BinaryIO) -> list[Document]:
    """Load a PDF from an uploaded file object."""

    filename = _get_filename(file)
    file_bytes = _read_uploaded_bytes(file)

    with NamedTemporaryFile(
        suffix=".pdf",
        prefix="rag_upload_",
        delete=True,
    ) as temp_file:

        temp_file.write(file_bytes)
        temp_file.flush()

        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()

    if not documents:
        raise ValueError(
            f"No readable content found in uploaded PDF: {filename}"
        )

    for document in documents:
        document.metadata.update(
            _build_metadata(
                filename=filename,
                file_type="pdf",
            )
        )

    return documents


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def _load_docx_from_bytes(
    file: BinaryIO,
) -> list[Document]:
    """Load a DOCX document."""

    filename = _get_filename(file)
    file_bytes = _read_uploaded_bytes(file)

    # document = DocxDocument(
    #     StringIO()
    # ) if False else None

    # python-docx requires a binary file-like object.
    from io import BytesIO

    docx_document = DocxDocument(
        BytesIO(file_bytes)
    )

    parts = []

    for paragraph in docx_document.paragraphs:
        text = paragraph.text.strip()

        if text:
            parts.append(text)

    # Include table content as structured text.
    for table_index, table in enumerate(
        docx_document.tables,
        start=1,
    ):
        parts.append(
            f"[Table {table_index}]"
        )

        for row in table.rows:
            cells = [
                cell.text.strip()
                for cell in row.cells
            ]

            if any(cells):
                parts.append(
                    " | ".join(cells)
                )

    content = "\n".join(parts).strip()

    if not content:
        raise ValueError(
            f"No readable content found in DOCX: {filename}"
        )

    return [
        Document(
            page_content=content,
            metadata=_build_metadata(
                filename=filename,
                file_type="docx",
            ),
        )
    ]


# ---------------------------------------------------------------------------
# TXT / Markdown
# ---------------------------------------------------------------------------

def _load_text_from_bytes(
    file: BinaryIO,
    file_type: str,
) -> list[Document]:
    """Load plain-text or Markdown content."""

    filename = _get_filename(file)
    file_bytes = _read_uploaded_bytes(file)

    try:
        content = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Unable to decode text file as UTF-8: {filename}"
            ) from error

    content = content.strip()

    if not content:
        raise ValueError(
            f"No readable content found in {filename}"
        )

    return [
        Document(
            page_content=content,
            metadata=_build_metadata(
                filename=filename,
                file_type=file_type,
            ),
        )
    ]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _load_csv_from_bytes(
    file: BinaryIO,
) -> list[Document]:
    """
    Load CSV data.

    Each row becomes a separate LangChain Document.
    The header is included with every row so retrieval retains
    column context.
    """

    filename = _get_filename(file)
    file_bytes = _read_uploaded_bytes(file)

    try:
        content = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Unable to decode CSV as UTF-8: {filename}"
            ) from error

    reader = csv.reader(StringIO(content))

    rows = list(reader)

    if not rows:
        raise ValueError(
            f"No readable rows found in CSV: {filename}"
        )

    header = rows[0]

    if not any(cell.strip() for cell in header):
        raise ValueError(
            f"CSV header is empty: {filename}"
        )

    documents = []

    for row_number, row in enumerate(
        rows[1:],
        start=2,
    ):
        if not any(cell.strip() for cell in row):
            continue

        values = []

        for column_index, value in enumerate(row):
            column_name = (
                header[column_index]
                if column_index < len(header)
                else f"column_{column_index + 1}"
            )

            values.append(
                f"{column_name}: {value}"
            )

        page_content = (
            f"CSV row {row_number}\n"
            + "\n".join(values)
        )

        documents.append(
            Document(
                page_content=page_content,
                metadata=_build_metadata(
                    filename=filename,
                    file_type="csv",
                    row=row_number,
                ),
            )
        )

    if not documents:
        raise ValueError(
            f"CSV contains no readable data rows: {filename}"
        )

    return documents

# ---------------------------------------------------------------------------
# XLSX
# ---------------------------------------------------------------------------

def _format_xlsx_value(value) -> str | None:
    """
    Convert a single spreadsheet cell into clean text.

    Returns None when the cell holds no usable content so empty
    cells never produce dangling labels in the document text.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return str(value)

    # numpy floats subclass float, so integral values such as
    # 2.0 are rendered as "2" instead of "2.0".
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

        return str(value)

    if isinstance(value, datetime):
        if (
            value.hour == 0
            and value.minute == 0
            and value.second == 0
            and value.microsecond == 0
        ):
            return value.strftime("%Y-%m-%d")

        return value.strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(value, (date, time)):
        return value.isoformat()

    text = str(value).strip()

    return text or None


def _normalize_xlsx_columns(columns) -> list[str]:
    """
    Build stable, unique column names for one worksheet.

    Blank header cells and pandas placeholder names such as
    "Unnamed: 1" are replaced with positional names so internal
    reader naming never leaks into the document text.
    """

    names = []
    used: set[str] = set()

    for position, column in enumerate(columns, start=1):

        raw_name = (
            ""
            if column is None
            else str(column).strip()
        )

        if (
            not raw_name
            or raw_name.lower().startswith("unnamed:")
        ):
            base_name = f"column_{position}"
        else:
            base_name = raw_name

        unique_name = base_name

        suffix = 2

        while unique_name in used:
            unique_name = (
                f"{base_name}_{suffix}"
            )
            suffix += 1

        used.add(unique_name)

        names.append(unique_name)

    return names


def _load_xlsx_from_bytes(
    file: BinaryIO,
) -> list[Document]:
    """
    Load an Excel workbook.

    Every non-empty row of every worksheet becomes a separate
    LangChain Document. Column names are repeated with each row,
    and the true worksheet row number is preserved in both the
    document text and the metadata so retrieval can cite the
    exact row even when the sheet contains blank rows.
    """

    filename = _get_filename(file)
    file_bytes = _read_uploaded_bytes(file)

    try:
        workbook = pd.read_excel(
            BytesIO(file_bytes),
            sheet_name=None,
            engine="openpyxl",
            header=0,
        )
    except Exception as error:
        raise ValueError(
            f"Unable to read Excel workbook: {filename}"
        ) from error

    if not workbook:
        raise ValueError(
            f"No worksheets found in Excel workbook: {filename}"
        )

    documents = []

    for sheet_name, dataframe in workbook.items():

        column_names = _normalize_xlsx_columns(
            dataframe.columns
        )

        # read_excel with header=0 keeps a positional index, so
        # index 0 is worksheet row 2. Blank rows are skipped at
        # emit time instead of being dropped, which keeps every
        # reported row number aligned with the real worksheet.
        for position, (_, row) in enumerate(
            dataframe.iterrows()
        ):

            row_number = position + 2

            values = []

            for column_name, value in zip(
                column_names,
                row.tolist(),
            ):

                formatted_value = _format_xlsx_value(
                    value
                )

                if formatted_value is None:
                    continue

                values.append(
                    f"{column_name}: {formatted_value}"
                )

            if not values:
                continue

            page_content = (
                f"Excel sheet: {sheet_name}\n"
                f"Excel row: {row_number}\n"
                + "\n".join(values)
            )

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=_build_metadata(
                        filename=filename,
                        file_type="xlsx",
                        sheet=sheet_name,
                        row=row_number,
                    ),
                )
            )

    if not documents:
        raise ValueError(
            f"No readable data found in Excel workbook: {filename}"
        )

    return documents


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------

def _load_pptx_from_bytes(
    file: BinaryIO,
) -> list[Document]:
    """Load a PowerPoint presentation, one Document per slide."""

    filename = _get_filename(file)
    file_bytes = _read_uploaded_bytes(file)

    from io import BytesIO

    presentation = Presentation(
        BytesIO(file_bytes)
    )

    documents = []

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1,
    ):
        slide_parts = []

        for shape in slide.shapes:
            if not hasattr(shape, "text"):
                continue

            text = shape.text.strip()

            if text:
                slide_parts.append(text)

        content = "\n".join(slide_parts).strip()

        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata=_build_metadata(
                    filename=filename,
                    file_type="pptx",
                    slide=slide_number,
                ),
            )
        )

    if not documents:
        raise ValueError(
            f"No readable text found in PPTX: {filename}"
        )

    return documents


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_documents(
    file_source: str | Path | BinaryIO,
) -> list[Document]:
    """
    Load a supported document into normalized LangChain Documents.

    Supported formats:

    - PDF
    - DOCX
    - TXT
    - Markdown
    - CSV
    - XLSX
    - PPTX

    The rest of the RAG pipeline only receives LangChain Documents,
    so it remains independent of the original file format.
    """

    filename = _get_filename(file_source)
    extension = _get_extension(file_source)

    if isinstance(file_source, (str, Path)):
        path = Path(file_source)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        if extension == ".pdf":
            return _load_pdf_from_path(path)

        file_bytes = path.read_bytes()

    else:
        file_bytes = _read_uploaded_bytes(file_source)

    # Reuse the same binary representation through a small
    # in-memory binary stream.
    from io import BytesIO

    binary_file = BytesIO(file_bytes)
    binary_file.name = filename

    if extension == ".pdf":
        return _load_pdf_from_upload(binary_file)

    if extension == ".docx":
        return _load_docx_from_bytes(binary_file)

    if extension == ".txt":
        return _load_text_from_bytes(
            binary_file,
            file_type="txt",
        )

    if extension == ".md":
        return _load_text_from_bytes(
            binary_file,
            file_type="markdown",
        )

    if extension == ".csv":
        return _load_csv_from_bytes(binary_file)

    if extension == ".xlsx":
        return _load_xlsx_from_bytes(binary_file)

    if extension == ".pptx":
        return _load_pptx_from_bytes(binary_file)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


def load_pdf(
    file_source: str | Path | BinaryIO,
) -> list[Document]:
    """
    Backward-compatible PDF-only loader.

    Existing tests and code using load_pdf() continue to work.
    """

    extension = _get_extension(file_source)

    if extension != ".pdf":
        raise ValueError(
            f"load_pdf() only accepts PDF files, received: {extension}"
        )

    return load_documents(file_source)