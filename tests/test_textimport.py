"""Tests for speech/textimport.py: cleaning, plain-text chapters, and EPUB import."""

import io
import zipfile

import pytest
import textimport as ti

from tests.epub_builder import make_epub, page

OPTIONS = ti.Options()


def chapters(book: ti.Book) -> list[str]:
    """Return the chapter titles of an imported book."""
    return [chapter["title"] for chapter in book.chapters]


def kinds(book: ti.Book) -> dict[str, int]:
    """Return each warning kind with its count."""
    return {w["kind"]: w["count"] for w in book.warnings}


# ---- cleaning ----


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("As shown [12], it works.", "As shown, it works."),
        ("Both [3, 4] and [5-7] agree.", "Both and agree."),
        ("It works (Smith et al., 2019).", "It works."),
        ("It works (Smith and Jones 2019; Lee, 2020a).", "It works."),
        ("In 1914 (the war) it began (born 1950).", "In 1914 (the war) it began (born 1950)."),
        ("See https://example.com/a?b=1. Then go.", "See. Then go."),
        ("Cited as doi:10.1000/xyz123, yes.", "Cited as, yes."),
    ],
    ids=[
        "numeric citation",
        "list and range",
        "author-year with et al",
        "several author-years",
        "ordinary parentheses stay",
        "web address",
        "doi",
    ],
)
def test_clean_line_removes_citations_and_links_and_closes_the_gap(line, expected):
    """Citations and links disappear with their leading space, and punctuation stays attached."""
    assert ti.clean_line(line, OPTIONS, ti.Warnings()) == expected


def test_clean_line_leaves_text_alone_when_an_option_is_off():
    """Each removal can be switched off."""
    off = ti.Options(citations=False, author_year=False, links=False)
    line = "A [1] b (Lee, 2020) c http://x.org d."

    assert ti.clean_line(line, off, ti.Warnings()) == line


def test_clean_line_counts_what_it_removes_with_the_first_sample():
    """The warning has a count per kind and the first match as its sample."""
    warnings = ti.Warnings()

    ti.clean_line("A [1] b [2] c.", OPTIONS, warnings)

    assert warnings.as_list() == [{"kind": "numeric citation", "count": 2, "sample": "[1]"}]


# ---- plain text ----


def test_import_text_adds_headings_to_chapter_lines_that_stand_alone():
    """A line such as ``Chapter 2`` between blank lines starts a chapter; prose does not."""
    text = (
        b"Preface words.\n\nChapter 1\n\nFirst words here.\n\n"
        b"He read chapter 2 aloud to them.\n\nCHAPTER II.\n\nSecond words.\n"
    )

    book = ti.import_text(text, OPTIONS)

    assert chapters(book) == ["Opening", "Chapter 1", "Chapter II"]
    assert "He read chapter 2 aloud to them." in book.text
    assert kinds(book)["chapter heading detected"] == 2


def test_import_text_joins_a_title_line_that_follows_a_chapter_number():
    """A short unpunctuated line after ``CHAPTER I.`` becomes part of its heading."""
    book = ti.import_text(b"CHAPTER I.\n\nDOWN THE HOLE\n\nAlice began to be tired.\n", OPTIONS)

    assert chapters(book) == ["Chapter I. Down The Hole"]
    assert "Alice began to be tired." in book.text


def test_import_text_keeps_existing_headings_and_skips_detection():
    """Text that already has ``#`` headings is not scanned for chapter lines."""
    book = ti.import_text(b"# Mine\n\nWords.\n\nChapter 5\n\nMore words.\n", OPTIONS)

    assert chapters(book) == ["Mine"]
    assert "Chapter 5" in book.text


def test_import_text_joins_hyphenated_line_breaks_and_drops_page_numbers():
    """A word split across lines is joined, and a line with only a page number is removed."""
    book = ti.import_text(b"The exper-\niment ran.\n\n42\n\nNext words.\n", OPTIONS)

    assert "The experiment ran." in book.text
    assert "\n42\n" not in book.text
    assert kinds(book)["page number removed"] == 1


def test_import_text_reads_windows_1252_and_rejects_binary():
    """Bytes that are not UTF-8 decode as Windows-1252, and a file with NUL bytes is refused."""
    assert "café" in ti.import_text("café words.".encode("cp1252"), OPTIONS).text
    with pytest.raises(ti.ImportFailureError, match="not text"):
        ti.import_text(b"abc\x00def", OPTIONS)


def test_import_document_refuses_pdf_and_picks_the_importer_by_content():
    """A PDF gets a clear message, a zip is read as an EPUB, and anything else is text."""
    with pytest.raises(ti.ImportFailureError, match="PDF"):
        ti.import_document(b"%PDF-1.7 ...", "paper.pdf", OPTIONS)
    epub = make_epub({"a.xhtml": page("<h1>One</h1><p>Words here.</p>")}, [("a.xhtml", "One")])

    assert chapters(ti.import_document(epub, "x.bin", OPTIONS)) == ["One"]
    assert "Plain words." in ti.import_document(b"Plain words.", "x.txt", OPTIONS).text


# ---- EPUB chapters ----


@pytest.mark.parametrize("kind", ["nav", "ncx"])
def test_import_epub_makes_a_chapter_for_each_table_of_contents_entry(kind):
    """Chapters follow the navigation document or the NCX, in order, with the book's title."""
    files = {
        "a.xhtml": page("<h1>One</h1><p>First words.</p>"),
        "b.xhtml": page("<h1>Two</h1><p>Second words.</p>"),
    }

    book = ti.import_epub(
        make_epub(files, [("a.xhtml", "One"), ("b.xhtml", "Two")], kind=kind), OPTIONS
    )

    assert book.title == "A Book"
    assert book.text == "# One\n\nFirst words.\n\n# Two\n\nSecond words.\n"
    assert [c["chars"] for c in book.chapters] == [12, 13]


def test_import_epub_cuts_chapters_at_anchors_inside_one_file():
    """Several chapters in one file start at their fragment ids."""
    body = (
        '<h2 id="c1">One</h2><p>First words.</p><p>More first.</p>'
        '<h2 id="c2">Two</h2><p>Second words.</p>'
    )

    book = ti.import_epub(
        make_epub({"all.xhtml": page(body)}, [("all.xhtml#c1", "One"), ("all.xhtml#c2", "Two")]),
        OPTIONS,
    )

    assert book.text == "# One\n\nFirst words.\n\nMore first.\n\n# Two\n\nSecond words.\n"


def test_import_epub_continues_a_chapter_over_several_files():
    """Files with no table of contents entry belong to the chapter before them."""
    files = {
        "a.xhtml": page("<h1>One</h1><p>First words.</p>"),
        "a2.xhtml": page("<p>Still chapter one.</p>"),
        "b.xhtml": page("<h1>Two</h1><p>Second words.</p>"),
    }

    book = ti.import_epub(make_epub(files, [("a.xhtml", "One"), ("b.xhtml", "Two")]), OPTIONS)

    assert book.text == "# One\n\nFirst words.\n\nStill chapter one.\n\n# Two\n\nSecond words.\n"


def test_import_epub_skips_text_before_the_first_entry_and_says_so():
    """A title page or dedication before chapter one is left out, with its size in a warning."""
    files = {
        "front.xhtml": page("<p>For my mother.</p>"),
        "a.xhtml": page("<h1>One</h1><p>First words.</p>"),
    }

    book = ti.import_epub(make_epub(files, [("a.xhtml", "One")]), OPTIONS)

    assert "mother" not in book.text
    assert kinds(book)["text before the first chapter skipped"] == len("For my mother.")


def test_import_epub_drops_an_entry_with_no_text_of_its_own():
    """A part page that only contains a title gives no chapter, and its chapters remain."""
    files = {
        "part.xhtml": page("<h1>Part One</h1>"),
        "a.xhtml": page("<h1>Chapter One</h1><p>First words.</p>"),
    }
    toc = [("part.xhtml", "Part One"), ("a.xhtml", "Chapter One")]

    assert chapters(ti.import_epub(make_epub(files, toc), OPTIONS)) == ["Chapter One"]


def test_import_epub_leaves_out_non_linear_files_and_non_body_sections():
    """Files marked non-linear, and sections typed as copyright or bibliography, are skipped."""
    files = {
        "a.xhtml": page("<h1>One</h1><p>First words.</p>"),
        "extra.xhtml": page("<p>Bonus words.</p>"),
        "legal.xhtml": page("<p>All rights reserved.</p>", ' epub:type="copyright-page"'),
        "refs.xhtml": page("<p>Smith, 2019.</p>", ' epub:type="bibliography"'),
    }

    book = ti.import_epub(
        make_epub(files, [("a.xhtml", "One")], nonlinear=("extra.xhtml",)), OPTIONS
    )

    assert book.text == "# One\n\nFirst words.\n"
    assert kinds(book)["section skipped"] == 2


def test_import_epub_uses_each_file_as_a_chapter_without_a_table_of_contents():
    """With no navigation, the first heading titles each file's chapter, else its number."""
    files = {
        "a.xhtml": page("<h1>Alpha</h1><p>First words.</p>"),
        "b.xhtml": page("<p>Second words.</p>"),
    }

    book = ti.import_epub(make_epub(files, [], kind="none"), OPTIONS)

    assert chapters(book) == ["Alpha", "Section 2"]
    assert "no table of contents" in kinds(book)


def test_import_epub_starts_at_the_top_of_the_file_when_an_anchor_is_missing():
    """An entry whose fragment does not exist starts its chapter at the start of the file."""
    files = {"a.xhtml": page("<p>First words.</p>")}

    book = ti.import_epub(make_epub(files, [("a.xhtml#gone", "One")]), OPTIONS)

    assert book.text == "# One\n\nFirst words.\n"


# ---- EPUB text ----


def test_import_epub_decodes_html_entities_the_xml_parser_would_reject():
    """Named entities such as nbsp and mdash are decoded, and a non-breaking space is a space."""
    files = {"a.xhtml": page("<h1>One</h1><p>Words&nbsp;here&mdash;and &amp; there.</p>")}

    book = ti.import_epub(make_epub(files, [("a.xhtml", "One")]), OPTIONS)

    assert "Words here\u2014and & there." in book.text


def test_import_epub_removes_note_markers_and_footnotes_but_not_other_links_or_exponents():
    """Footnote links, symbol superscripts and note bodies go; linked words and exponents stay."""
    body = (
        '<h1>One</h1><p>A claim<sup><a href="#n1" epub:type="noteref">1</a></sup> and '
        'another<a href="#n2">[2]</a> and a <a href="b.xhtml">linked word</a> and 10<sup>3</sup> '
        "and starred<sup>*</sup>.</p>"
        '<aside epub:type="footnote" id="n1"><p>The note itself.</p></aside>'
    )

    book = ti.import_epub(make_epub({"a.xhtml": page(body)}, [("a.xhtml", "One")]), OPTIONS)

    assert "A claim and another and a linked word and 103 and starred." in book.text
    assert "note itself" not in book.text
    assert kinds(book)["footnote skipped"] == 1


def test_import_epub_skips_tables_and_figures_and_reads_equations_by_their_alt_text():
    """Tables and figures are left out with a count; math is read from its alttext."""
    body = (
        "<h1>One</h1><table><tr><td>cell</td></tr></table>"
        "<figure><img src='x.png'/><figcaption>A chart.</figcaption></figure>"
        '<p>It follows that <math alttext="x squared"><mi>x</mi></math> grows.</p>'
    )

    book = ti.import_epub(make_epub({"a.xhtml": page(body)}, [("a.xhtml", "One")]), OPTIONS)

    assert "cell" not in book.text
    assert "chart" not in book.text
    assert "It follows that x squared grows." in book.text
    assert (
        kinds(book)["table skipped"],
        kinds(book)["figure skipped"],
        kinds(book)["equation"],
    ) == (1, 1, 1)


def test_import_epub_reads_figure_captions_when_asked():
    """With captions on, the caption text is a paragraph."""
    body = "<h1>One</h1><figure><img src='x.png'/><figcaption>A chart.</figcaption></figure>"

    book = ti.import_epub(
        make_epub({"a.xhtml": page(body)}, [("a.xhtml", "One")]), ti.Options(captions=True)
    )

    assert "A chart." in book.text


def test_import_epub_ends_headings_and_list_items_with_a_period():
    """A heading or list item with no final punctuation receives one, so chunks do not merge."""
    body = "<h1>One</h1><h2>Methods</h2><ul><li>First item</li><li>Second item.</li></ul>"

    book = ti.import_epub(make_epub({"a.xhtml": page(body)}, [("a.xhtml", "One")]), OPTIONS)

    assert book.text == "# One\n\nMethods.\n\nFirst item.\n\nSecond item.\n"


def test_import_epub_cannot_start_a_chapter_from_a_body_line_that_looks_like_a_heading():
    """Body text that begins with ``# `` loses the hashes, so it stays inside its chapter."""
    files = {"a.xhtml": page("<h1>One</h1><p># Not a heading</p><p>Words.</p>")}

    book = ti.import_epub(make_epub(files, [("a.xhtml", "One")]), OPTIONS)

    assert chapters(book) == ["One"]
    assert "\nNot a heading\n" in book.text


def test_import_epub_cleans_a_chapter_title_to_one_line_without_hashes():
    """A table of contents label with a hash or a line break becomes a plain one-line title."""
    files = {"a.xhtml": page("<p>Words here.</p>")}

    book = ti.import_epub(make_epub(files, [("a.xhtml", "## Big\n  Idea")]), OPTIONS)

    assert chapters(book) == ["Big Idea"]


def test_import_epub_removes_citations_from_chapter_text():
    """The cleaning options apply to EPUB paragraphs too."""
    files = {"a.xhtml": page("<h1>One</h1><p>It works [4] (Lee et al., 2020).</p>")}

    book = ti.import_epub(make_epub(files, [("a.xhtml", "One")]), OPTIONS)

    assert "It works." in book.text


# ---- EPUB limits and damage ----


def test_import_epub_rejects_files_that_are_not_epubs():
    """A zip without a container file, a corrupt zip, and a bad package all raise with a message."""
    plain_zip = io.BytesIO()
    with zipfile.ZipFile(plain_zip, "w") as archive:
        archive.writestr("a.txt", "hello")

    with pytest.raises(ti.ImportFailureError, match="container.xml"):
        ti.import_epub(plain_zip.getvalue(), OPTIONS)
    with pytest.raises(ti.ImportFailureError, match="not a valid"):
        ti.import_epub(b"PK\x03\x04 not really a zip", OPTIONS)


def test_import_epub_rejects_an_archive_with_too_many_entries_or_too_much_data(monkeypatch):
    """Entry count and expanded size are limited before anything is parsed."""
    epub = make_epub({"a.xhtml": page("<h1>One</h1><p>Words.</p>")}, [("a.xhtml", "One")])

    monkeypatch.setattr(ti, "MAX_ENTRIES", 3)
    with pytest.raises(ti.ImportFailureError, match="entries"):
        ti.import_epub(epub, OPTIONS)
    monkeypatch.setattr(ti, "MAX_ENTRIES", 5000)
    monkeypatch.setattr(ti, "MAX_TOTAL_BYTES", 100)
    with pytest.raises(ti.ImportFailureError, match="expands"):
        ti.import_epub(epub, OPTIONS)


def test_import_epub_reports_a_content_file_over_the_entry_limit_as_missing(monkeypatch):
    """One oversized file is left out with a warning, instead of being read into memory."""
    files = {
        "a.xhtml": page("<h1>One</h1><p>First words.</p>"),
        "big.xhtml": page("<h1>Two</h1><p>" + "x" * 5000 + "</p>"),
    }
    epub = make_epub(files, [("a.xhtml", "One"), ("big.xhtml", "Two")])
    monkeypatch.setattr(ti, "MAX_ENTRY_BYTES", 3000)

    book = ti.import_epub(epub, OPTIONS)

    assert chapters(book) == ["One"]
    assert kinds(book)["missing file"] == 1


def test_import_epub_warns_when_the_text_is_over_the_job_limit(monkeypatch):
    """The importer says so when the job would refuse the text."""
    monkeypatch.setattr(ti, "MAX_JOB_CHARS", 10)
    files = {"a.xhtml": page("<h1>One</h1><p>Words that run long.</p>")}

    book = ti.import_epub(make_epub(files, [("a.xhtml", "One")]), OPTIONS)

    assert "over the job limit" in kinds(book)
