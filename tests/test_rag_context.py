"""Tests for document parsing, structure-aware chunking, and Ukrainian hybrid RAG."""

from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument

import context_loader


def _build_minimal_pdf(text: str) -> bytes:
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("latin1")
    objs = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
        ),
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        (
            f"5 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream\nendobj\n"
        ),
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj in objs:
        offsets.append(len(out))
        out.extend(obj)
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(out)


def test_parse_txt_code_docx_pdf_and_vocab(tmp_path: Path) -> None:
    txt_p = tmp_path / "notes.txt"
    txt_p.write_text("Тема захисту: AI Copilot через WASAPI та CUDA", encoding="utf-8")

    py_p = tmp_path / "algo.py"
    py_p.write_text("def dijkstra_search(graph):\n    return len(graph)\n", encoding="utf-8")

    docx_p = tmp_path / "report.docx"
    doc = DocxDocument()
    doc.add_paragraph("Розділ 1. Архітектура FastAPI та PostgreSQL.")
    tbl = doc.add_table(rows=1, cols=2)
    tbl.cell(0, 0).text = "Метрика"
    tbl.cell(0, 1).text = "WER 4.2%"
    doc.save(str(docx_p))

    pdf_p = tmp_path / "slides.pdf"
    pdf_p.write_bytes(_build_minimal_pdf("Slide 1: Real-time Whisper Architecture"))

    cm = context_loader.ContextManager()
    loaded, errs = cm.add_files([txt_p, py_p, docx_p, pdf_p])
    assert not errs
    assert len(loaded) == 4
    vocab = cm.extract_vocabulary_terms()
    assert "WASAPI" in vocab and "CUDA" in vocab and "FastAPI" in vocab


def test_structure_aware_chunking_preserves_sections() -> None:
    doc = context_loader.LoadedDocument(
        path="report.md",
        name="report.md",
        content=(
            "# Вступ\n"
            + ("Загальний опис системи. " * 40)
            + "\n\n# Алгоритм Дейкстри\n"
            + ("Складність алгоритму з купою Фібоначчі дорівнює O(E + V log V). " * 25)
        ),
        char_count=2000,
    )
    chunks = context_loader.chunk_document(doc, target_chars=600, overlap_chars=100)
    assert len(chunks) >= 3
    sections = {c.section for c in chunks}
    assert "Вступ" in sections
    assert "Алгоритм Дейкстри" in sections


def test_tiered_cag_vs_hybrid_rag_ukrainian_retrieval_and_invalidation(tmp_path: Path) -> None:
    f1 = tmp_path / "01_intro.md"
    f1.write_text(
        "# Вступ\n" + ("Проєкт присвячено інтерфейсу користувача та стилізації вікон. " * 60),
        encoding="utf-8",
    )
    f2 = tmp_path / "02_database.md"
    f2.write_text(
        "# База даних\n"
        "Для зберігання транзакцій використано PostgreSQL з B-tree індексами та ACID-ізоляцією.\n"
        + ("Резервне копіювання виконується щогодини. " * 60),
        encoding="utf-8",
    )
    f3 = tmp_path / "03_ml.md"
    f3.write_text(
        "# Нейронна мережа\n"
        "Розпізнавання мовлення базується на трансформері faster-whisper large-v3-turbo на CUDA.\n"
        + ("Обробка аудіопотоку триває менше секунди. " * 60),
        encoding="utf-8",
    )

    cm = context_loader.ContextManager()
    cm.add_files([f1, f2, f3])
    rev1, hash1 = cm.content_revision
    assert rev1 == 1

    # Tier 1 CAG when threshold >= total_chars
    cag_ctx, cag_sources = cm.retrieve_context(
        query="Яка БД?", max_chars=25000, full_context_threshold=25000
    )
    assert cag_sources == ["01_intro.md", "02_database.md", "03_ml.md"]
    assert "PostgreSQL" in cag_ctx and "faster-whisper" in cag_ctx

    # Tier 2 Hybrid RAG when corpus > full_context_threshold and budget is tight
    rag_ctx, rag_sources = cm.retrieve_context(
        query="Які індекси та властивості бази даних PostgreSQL використовуються?",
        max_chars=1800,
        full_context_threshold=1000,
    )
    assert "PostgreSQL" in rag_ctx and "B-tree" in rag_ctx
    assert any("02_database.md" in s for s in rag_sources)

    # Deletion propagates immediately into index and content revision
    assert cm.remove_file(str(f2)) is True
    rev2, hash2 = cm.content_revision
    assert rev2 > rev1 and hash2 != hash1
    rag_after_delete, _ = cm.retrieve_context(
        query="PostgreSQL B-tree",
        max_chars=1800,
        full_context_threshold=1000,
    )
    assert "PostgreSQL" not in rag_after_delete


def test_csharp_section_chunking_and_request_budget_caps() -> None:
    import config
    import main

    doc = context_loader.LoadedDocument(
        path="OrderService.cs",
        name="OrderService.cs",
        content=(
            "using System;\n\n"
            "namespace DemoApp.Services\n"
            "{\n"
            "    public class OrderService\n"
            "    {\n"
            "        public void Process() { }\n"
            "    }\n\n"
            "    public interface IOrderRepo\n"
            "    {\n"
            "        int Count();\n"
            "    }\n\n"
            "    public record OrderDto(int Id, string Name);\n\n"
            "    public enum OrderStatus { New, Paid }\n"
            "}\n"
        ),
        char_count=320,
    )
    chunks = context_loader.chunk_document(doc, target_chars=600, overlap_chars=100)
    sections = {c.section for c in chunks}
    assert "Код: DemoApp.Services" in sections
    assert "Код: OrderService" in sections
    assert "Код: IOrderRepo" in sections
    assert "Код: OrderDto" in sections
    assert "Код: OrderStatus" in sections

    groq_cfg = config.AppConfig(
        llm_provider="groq",
        context_max_chars=120000,
        rag_context_budget_chars=24000,
    )
    assert main.compute_request_context_budget(groq_cfg, "concise") == (6500, 5000)
    assert main.compute_request_context_budget(groq_cfg, "detailed") == (10000, 7500)

    gemini_cfg = config.AppConfig(
        llm_provider="gemini",
        context_max_chars=120000,
        rag_context_budget_chars=12000,
    )
    assert main.compute_request_context_budget(gemini_cfg, "concise") == (12000, 9000)


def test_folder_expansion_docx_table_dedup_reload_and_sources_budget(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    (proj / "bin").mkdir(parents=True)
    (proj / "node_modules" / "pkg").mkdir(parents=True)
    (proj / "src").mkdir(parents=True)

    (proj / "bin" / "Build.cs").write_text("class IgnoredBuild {}", encoding="utf-8")
    (proj / "node_modules" / "pkg" / "index.js").write_text("console.log(1);", encoding="utf-8")
    (proj / "src" / "Form1.Designer.cs").write_text("partial class Form1 {}", encoding="utf-8")
    good_cs = proj / "src" / "PaymentService.cs"
    good_cs.write_text(
        "[Authorize]\npublic class PaymentService\n{\n"
        "    [HttpPost]\n    public void Charge() { }\n}\n",
        encoding="utf-8",
    )
    notes = proj / "notes.md"
    notes.write_text("# Архітектура\nБаза даних зберігає транзакції.\n", encoding="utf-8")

    # DOCX with headings + horizontally merged cells
    docx_file = proj / "report.docx"
    d = DocxDocument()
    d.add_heading("Вступ", level=1)
    d.add_paragraph("Опис сервісу оплати.")
    tbl = d.add_table(rows=1, cols=3)
    merged = tbl.cell(0, 0).merge(tbl.cell(0, 2))
    merged.text = "Об'єднана комірка"
    d.save(str(docx_file))

    expanded, warnings = context_loader.expand_context_paths([str(proj)])
    rel_names = {label for _p, label in expanded}
    assert any("PaymentService.cs" in n for n in rel_names)
    assert any("notes.md" in n for n in rel_names)
    assert not any("Build.cs" in n for n in rel_names)
    assert not any("index.js" in n for n in rel_names)
    assert not any("Designer.cs" in n for n in rel_names)
    assert len(warnings) == 1 and "пропущено" in warnings[0].lower()

    cm = context_loader.ContextManager()
    loaded, errs = cm.add_files(expanded)
    assert not errs and len(loaded) >= 3

    docx_doc = next(doc for doc in cm.documents if doc.name == "report.docx")
    assert "# Вступ" in docx_doc.content
    assert docx_doc.content.count("Об'єднана комірка") == 1

    # On-disk modification detected by reload_modified_files()
    good_cs.write_text(
        "[Authorize]\npublic class PaymentService\n{\n"
        "    public void RefundPayment() { }\n}\n",
        encoding="utf-8",
    )
    reloaded, r_errs = cm.reload_modified_files()
    assert not r_errs
    assert any(doc.name == "PaymentService.cs" for doc in reloaded)

    # Oversized first file in full_context mode must NOT falsely claim skipped files in sources
    big_a = tmp_path / "big_a.txt"
    big_b = tmp_path / "big_b.txt"
    big_a.write_text("A " * 800, encoding="utf-8")
    big_b.write_text("B " * 400, encoding="utf-8")
    cm2 = context_loader.ContextManager()
    cm2.add_files([str(big_a), str(big_b)])
    _ctx, srcs = cm2.retrieve_context(query="test", max_chars=500, full_context_threshold=5000)
    assert len(srcs) == 1 and "big_a.txt" in srcs[0]
    assert all("big_b.txt" not in s for s in srcs)


def test_loaded_document_metadata_and_folder_artifact_filtering(tmp_path: Path) -> None:
    pdf_a = tmp_path / "dir_a" / "slides.pdf"
    pdf_b = tmp_path / "dir_b" / "slides.pdf"
    pdf_a.parent.mkdir(parents=True)
    pdf_b.parent.mkdir(parents=True)
    pdf_a.write_bytes(_build_minimal_pdf("Page 1 architecture"))
    pdf_b.write_bytes(_build_minimal_pdf("Page 1 database"))

    (tmp_path / "dir_a" / "App.deps.json").write_text('{"targets":{}}', encoding="utf-8")
    (tmp_path / "dir_a" / "App.runtimeconfig.json").write_text('{"runtimeOptions":{}}', encoding="utf-8")
    (tmp_path / "dir_a" / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")

    expanded, notices = context_loader.expand_context_paths([str(tmp_path)])
    names = {p.name for p, _ in expanded}
    assert "slides.pdf" in names
    assert "App.deps.json" not in names
    assert "App.runtimeconfig.json" not in names
    assert "package-lock.json" not in names
    assert any("пропущено" in n.lower() for n in notices)

    cm = context_loader.ContextManager()
    loaded, errs = cm.add_files([str(pdf_a), str(pdf_b)])
    assert not errs and len(loaded) == 2
    for doc in cm.documents:
        assert doc.token_estimate >= 1
        assert doc.page_count == 1
        assert doc.truncated is False
        assert "/" in doc.label
