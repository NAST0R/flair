"""Estrazione testo dai formati documento più diffusi — SOLO stdlib.

DOCX/XLSX/PPTX/ODT sono ZIP+XML e si estraggono con `zipfile` + ElementTree a
qualità piena, in stdlib. Il PDF passa da `pypdf` (puro Python, zero dipendenze
transitive): il formato è un campo minato di casi limite — un parser fatto a
mano qui è crashato sul campo al primo PDF vero (escape ottali) — e reinventare
problemi già risolti non vale la candela. Resta il CANCELLO DI QUALITÀ: i PDF
scansionati/solo-immagine o dall'output inaffidabile producono un errore onesto
e azionabile invece di spazzatura nel contesto del modello. L'estrazione
ALIMENTA la pipeline di read_file (offset/limit/budget/header onesto): qui si
produce solo il testo.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from ..core.tool import ToolError

DOCUMENT_EXTS = {".docx", ".xlsx", ".pptx", ".odt", ".pdf"}


def extract_text(p: Path) -> tuple[str, str]:
    """Ritorna (testo, etichetta formato). Solleva ToolError con messaggio
    azionabile quando l'estrazione affidabile non è possibile."""
    ext = p.suffix.lower()
    try:
        if ext == ".docx":
            return _from_docx(p), "DOCX"
        if ext == ".odt":
            return _from_odt(p), "ODT"
        if ext == ".pptx":
            return _from_pptx(p), "PPTX"
        if ext == ".xlsx":
            return _from_xlsx(p), "XLSX"
        if ext == ".pdf":
            return _from_pdf(p), "PDF"
    except ToolError:
        raise
    except Exception as exc:
        # Rete al confine: l'estrazione è best-effort PER DESIGN, quindi qualunque
        # imprevisto (file malformato, caso limite del formato) deve degradare in
        # un errore di tool onesto — mai in uno stack trace (lezione di campo).
        raise ToolError(
            f"Cannot extract text from {p.name}: not a valid {ext[1:].upper()} "
            f"or an unsupported variant ({type(exc).__name__})."
        ) from exc
    raise ToolError(f"Unsupported document format: {ext}")


def _tag(el) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _from_docx(p: Path) -> str:
    with zipfile.ZipFile(p) as z:
        root = ElementTree.fromstring(z.read("word/document.xml"))
    out: list[str] = []
    for para in root.iter():
        if _tag(para) == "p":
            line = "".join(t.text or "" for t in para.iter() if _tag(t) == "t")
            out.append(line)
    return "\n".join(out).strip() or _empty(p)


def _from_odt(p: Path) -> str:
    with zipfile.ZipFile(p) as z:
        root = ElementTree.fromstring(z.read("content.xml"))
    out = ["".join(el.itertext()) for el in root.iter() if _tag(el) in ("p", "h")]
    return "\n".join(out).strip() or _empty(p)


def _from_pptx(p: Path) -> str:
    with zipfile.ZipFile(p) as z:
        slides = sorted(
            (n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.search(r"(\d+)", n).group(1)),   # type: ignore[union-attr]
        )
        out: list[str] = []
        for i, name in enumerate(slides, 1):
            root = ElementTree.fromstring(z.read(name))
            texts = [t.text or "" for t in root.iter() if _tag(t) == "t"]
            out.append(f"── slide {i} ──")
            out.extend(x for x in texts if x.strip())
    return "\n".join(out).strip() or _empty(p)


def _from_xlsx(p: Path) -> str:
    with zipfile.ZipFile(p) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            sroot = ElementTree.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(si.itertext()) for si in sroot if _tag(si) == "si"]
        names: list[str] = []
        if "xl/workbook.xml" in z.namelist():
            wb = ElementTree.fromstring(z.read("xl/workbook.xml"))
            names = [s.get("name", "") for s in wb.iter() if _tag(s) == "sheet"]
        sheets = sorted(
            (n for n in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)),
            key=lambda n: int(re.search(r"(\d+)", n).group(1)),   # type: ignore[union-attr]
        )
        out: list[str] = []
        for i, name in enumerate(sheets):
            label = names[i] if i < len(names) else f"sheet {i + 1}"
            out.append(f"── {label} ──")
            root = ElementTree.fromstring(z.read(name))
            for row in root.iter():
                if _tag(row) != "row":
                    continue
                cells: list[str] = []
                for c in row:
                    v = next((el for el in c.iter() if _tag(el) in ("v", "t")), None)
                    val = (v.text or "") if v is not None else ""
                    if c.get("t") == "s" and val.isdigit() and int(val) < len(shared):
                        val = shared[int(val)]
                    cells.append(val)
                if any(x.strip() for x in cells):
                    out.append(" | ".join(x.strip() for x in cells).rstrip(" |"))
    return "\n".join(out).strip() or _empty(p)


# ── PDF (pypdf) ───────────────────────────────────────────────────────────────

def _from_pdf(p: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:                     # ambiente incompleto: dillo chiaro
        raise ToolError(
            "PDF support needs the `pypdf` package (a flair dependency): "
            "reinstall with `pip install -e .` or `pip install pypdf`."
        ) from exc
    reader = PdfReader(p)
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):             # molti PDF: cifrati con password vuota
                raise ToolError("empty-password decrypt failed")
        except ToolError:
            raise ToolError(f"{p.name} is an encrypted PDF: decrypt it (or export to text) first.") from None
        except Exception as exc:
            raise ToolError(f"{p.name} is an encrypted PDF: decrypt it (or export to text) first.") from exc
    pages: list[str] = []
    for i, page in enumerate(reader.pages, 1):
        t = (page.extract_text() or "").strip()
        if t:
            pages.append(f"── page {i} ──\n{t}")
    text = "\n".join(pages).strip()
    _pdf_quality_gate(p, text)
    return text


def _pdf_quality_gate(p: Path, text: str) -> None:
    """I PDF scansionati o a font CID producono stringhe di glifi, non testo:
    meglio un errore onesto che spazzatura plausibile nel contesto del modello."""
    words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", text)
    good = sum(ch.isalnum() or ch.isspace() or ch in ".,;:!?'\"()[]{}|/\\-–—_*#@%&+=<>~`^$" for ch in text)
    if len(text) < 20 or not words or len(words) < 3 or good / max(1, len(text)) < 0.75:
        raise ToolError(
            f"Cannot reliably extract text from {p.name}: it looks scanned, image-only "
            "or CID-encoded (beyond the built-in extractor). Convert it to text/markdown "
            "first (e.g. with a PDF tool), then read the converted file."
        )


def _empty(p: Path) -> str:
    raise ToolError(f"No extractable text found in {p.name} (empty or unsupported layout).")
