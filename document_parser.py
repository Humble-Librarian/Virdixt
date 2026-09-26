"""
Document Ingestion & Multi-Format Parser
Extracts text and embedded images from PDF, DOCX, and TXT files.
Designed for both text-only filings and multimodal corporate reports.
"""

import os
import io
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParsedDocument:
    raw_text: str
    image_paths: List[str] = field(default_factory=list)
    file_type: str = "unknown"
    page_count: int = 1
    has_visuals: bool = False
    temp_dir: Optional[str] = None

    def cleanup(self):
        """Removes extracted temporary image files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception:
                pass


class DocumentParser:
    """Universal parser for PDF, DOCX, and TXT financial reports."""

    @staticmethod
    def parse_txt(file_path: str) -> ParsedDocument:
        """Parses plain text with multi-encoding fallback."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        content = ""
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, OSError):
                continue

        return ParsedDocument(
            raw_text=content.strip(),
            image_paths=[],
            file_type="TXT",
            page_count=1,
            has_visuals=False
        )

    @staticmethod
    def parse_pdf(file_path: str, extract_images: bool = True) -> ParsedDocument:
        """Parses PDF documents, extracting page text and embedded images/charts."""
        text_chunks = []
        image_paths = []
        temp_dir = tempfile.mkdtemp(prefix="virdixt_pdf_")
        page_count = 0

        # Method A: PyMuPDF (fitz) - ultra-fast text and high-res image extraction
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            page_count = len(doc)
            
            for page_num in range(page_count):
                page = doc[page_num]
                text_chunks.append(page.get_text())

                if extract_images:
                    image_list = page.get_images(full=True)
                    for img_index, img_info in enumerate(image_list):
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Filter tiny icons / bullets (keep images > 5KB / > 100x100)
                        if len(image_bytes) > 300:
                            img_filename = f"page_{page_num+1}_img_{img_index+1}.{image_ext}"
                            out_path = os.path.join(temp_dir, img_filename)
                            with open(out_path, "wb") as img_file:
                                img_file.write(image_bytes)
                            image_paths.append(out_path)

            doc.close()
            full_text = "\n\n".join([t.strip() for t in text_chunks if t.strip()])
            return ParsedDocument(
                raw_text=full_text,
                image_paths=image_paths,
                file_type="PDF",
                page_count=page_count,
                has_visuals=len(image_paths) > 0,
                temp_dir=temp_dir
            )
        except ImportError:
            pass

        # Method B: pypdf fallback
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            page_count = len(reader.pages)
            for page_idx, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    text_chunks.append(extracted)
                if extract_images and hasattr(page, "images"):
                    for img_idx, img in enumerate(page.images):
                        if len(img.data) > 300:
                            img_path = os.path.join(temp_dir, f"page_{page_idx+1}_img_{img_idx+1}.png")
                            with open(img_path, "wb") as f:
                                f.write(img.data)
                            image_paths.append(img_path)

            full_text = "\n\n".join(text_chunks)
            return ParsedDocument(
                raw_text=full_text,
                image_paths=image_paths,
                file_type="PDF",
                page_count=page_count,
                has_visuals=len(image_paths) > 0,
                temp_dir=temp_dir
            )
        except Exception as e:
            return ParsedDocument(
                raw_text=f"Error extracting PDF: {e}",
                file_type="PDF_ERROR",
                temp_dir=temp_dir
            )

    @staticmethod
    def parse_docx(file_path: str, extract_images: bool = True) -> ParsedDocument:
        """Parses DOCX documents, extracting paragraphs, tables, and embedded images from zip archive."""
        text_chunks = []
        image_paths = []
        temp_dir = tempfile.mkdtemp(prefix="virdixt_docx_")

        # 1. Extract text and tables via python-docx
        try:
            import docx
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text_chunks.append(para.text.strip())
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                    if row_text:
                        text_chunks.append(row_text)
        except Exception:
            pass

        # 2. Extract embedded images and charts directly from word/media zip archive
        if extract_images and zipfile.is_zipfile(file_path):
            try:
                with zipfile.ZipFile(file_path, 'r') as z:
                    for filename in z.namelist():
                        if filename.startswith('word/media/'):
                            image_data = z.read(filename)
                            if len(image_data) > 300:
                                base_name = os.path.basename(filename)
                                out_path = os.path.join(temp_dir, base_name)
                                with open(out_path, 'wb') as f:
                                    f.write(image_data)
                                image_paths.append(out_path)
            except Exception:
                pass

        full_text = "\n\n".join(text_chunks)
        return ParsedDocument(
            raw_text=full_text,
            image_paths=image_paths,
            file_type="DOCX",
            page_count=1,
            has_visuals=len(image_paths) > 0,
            temp_dir=temp_dir
        )

    @classmethod
    def parse(cls, file_path: str, extract_images: bool = True) -> ParsedDocument:
        """Auto-detects format and parses PDF, DOCX, or TXT document."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return cls.parse_pdf(file_path, extract_images=extract_images)
        elif ext in [".docx", ".doc"]:
            return cls.parse_docx(file_path, extract_images=extract_images)
        elif ext in [".txt", ".md", ".json", ".csv", ".log"]:
            return cls.parse_txt(file_path)
        else:
            # Fallback attempt as plain text
            return cls.parse_txt(file_path)
