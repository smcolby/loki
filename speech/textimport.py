"""Import books and papers as the text a narration job reads.

An importer returns text in the job's chapter format: a line ``# Chapter title`` starts a
chapter, and blank lines separate paragraphs. The Studio shows that text for review before
a job is created, so a wrong chapter split or a leftover page header can be fixed by hand.

Plain text with ``#`` headings is imported as written; without any, lines such as
``Chapter 12`` become headings.
An EPUB is read with the standard library only: chapters are cut at the anchors of its table
of contents, and footnote markers, tables, figures, equations and non-body sections are left
out. Every removal is reported as a warning with a count, so the review step shows what
changed.
"""

import io
import posixpath
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import unquote
from xml.etree import ElementTree as ET

from narration import HEADING, MAX_CHAPTERS, MAX_JOB_CHARS, split_chapters

MAX_ENTRIES = 5000
MAX_ENTRY_BYTES = 30 * 1024**2
MAX_TOTAL_BYTES = 200 * 1024**2
MAX_TITLE_CHARS = 120

NUMERIC_CITATION = re.compile(r"\s*\[\s*\d+(?:\s*[,–—-]\s*\d+)*\s*\]")
AUTHOR_YEAR_CITATION = re.compile(
    r"\s*\((?:[A-Z][\w'’-]+(?:\s+(?:and|&)\s+[A-Z][\w'’-]+)?(?:\s+et al\.?)?,?\s+"
    r"(?:1[5-9]|20)\d\d[a-z]?(?:,\s*(?:pp?\.\s*)?\d+(?:[-–]\d+)?)?;?\s*)+\)"
)
LINK = re.compile(r"(?:https?://|www\.|doi:)[^\s<>]+?(?=[.,;:!?)\]]*(?:\s|$))", re.IGNORECASE)
PAGE_NUMBER = re.compile(r"^\W*(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?\W*$", re.IGNORECASE)
ROMAN = re.compile(r"[ivxlcdm]+", re.IGNORECASE)
CHAPTER_LINE = re.compile(
    r"^(?:chapter|part|book|section)\s+(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven"
    r"|eight|nine|ten|eleven|twelve)\b[.:]?(?:\s+\S.{0,60})?$",
    re.IGNORECASE,
)
ROMAN_LINE = re.compile(r"^[IVXLCDM]{1,8}\.?$")
HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-z])")
NOTE_MARKER = re.compile(r"^[\[(]?(?:\d{1,3}|[*\u2020\u2021\u00a7]+)[\])]?$")
ENDS_A_SENTENCE = re.compile(r"[.!?:;\"'”’)]$")

# Section types (epub:type or ARIA role) that are not body text
SKIP_TYPES = frozenset(
    {
        "cover",
        "toc",
        "landmarks",
        "copyright-page",
        "titlepage",
        "index",
        "loi",
        "lot",
        "endnotes",
        "footnotes",
        "bibliography",
    }
)
NOTE_TYPES = frozenset({"footnote", "endnote", "rearnote", "note"})
SKIP_TAGS = frozenset({"script", "style", "head", "title", "rt", "rp", "table", "nav", "svg"})
VOID_TAGS = frozenset(
    {"br", "img", "hr", "meta", "link", "input", "col", "wbr", "area", "base", "source", "track"}
)
BLOCK_TAGS = frozenset(
    {"p", "div", "li", "blockquote", "pre", "section", "article", "dd", "dt", "figcaption"}
    | {"h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "body", "header", "footer", "tr"}
)
HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


class ImportFailureError(ValueError):
    """The uploaded file cannot be imported; the message says why."""


@dataclass(frozen=True)
class Options:
    """What the importers remove from the text.

    Parameters
    ----------
    citations : bool
        Remove bracketed numeric citations such as ``[12]`` and ``[3, 4]``.
    author_year : bool
        Remove parenthetical author-year citations such as ``(Smith et al., 2019)``.
    links : bool
        Remove web addresses and DOIs.
    captions : bool
        Read figure captions (EPUB). Figures are left out when this is False.
    """

    citations: bool = True
    author_year: bool = True
    links: bool = True
    captions: bool = False


class Warnings:
    """Counts what an importer changed, with the first example of each kind."""

    def __init__(self) -> None:
        self._found: dict[str, list[Any]] = {}

    def add(self, kind: str, sample: str = "", count: int = 1) -> None:
        """Record ``count`` more changes of one kind, keeping the first sample."""
        entry = self._found.setdefault(kind, [0, sample.strip()[:80]])
        entry[0] += count

    def as_list(self) -> list[dict[str, Any]]:
        """Return ``{"kind", "count", "sample"}`` for each kind, in the order first seen."""
        return [{"kind": k, "count": v[0], "sample": v[1]} for k, v in self._found.items()]


@dataclass(frozen=True)
class Book:
    """An imported document: its title, chapter-format text, chapter sizes, and warnings."""

    title: str
    text: str
    chapters: list[dict[str, Any]]
    warnings: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        """Return the fields as a JSON-ready dictionary."""
        return {
            "title": self.title,
            "text": self.text,
            "chapters": self.chapters,
            "warnings": self.warnings,
        }


def clean_line(line: str, options: Options, warnings: Warnings) -> str:
    """Remove citations and links from one line of text, and close the gaps they leave.

    Parameters
    ----------
    line : str
        A line or paragraph of body text.
    options : Options
        Which removals are on.
    warnings : Warnings
        Receives a count and a sample for each kind removed.

    Returns
    -------
    str
        The cleaned line.

    Examples
    --------
    >>> clean_line("As shown [12], it works (Smith et al., 2019).", Options(), Warnings())
    'As shown, it works.'
    """
    for kind, enabled, pattern in (
        ("numeric citation", options.citations, NUMERIC_CITATION),
        ("author-year citation", options.author_year, AUTHOR_YEAR_CITATION),
        ("link", options.links, LINK),
    ):
        match = pattern.search(line) if enabled else None
        if match:
            line, count = pattern.subn("", line)
            warnings.add(kind, match.group(0), count)
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"\s+([,.;:!?])", r"\1", line)).strip()


def _title_case(text: str) -> str:
    """Return ALL CAPS text in sentence-style capitals so it is not spelled out letter by letter."""
    if not text.isupper():
        return text
    return " ".join(w if ROMAN.fullmatch(w.rstrip(".")) else w.capitalize() for w in text.split())


def _detect_headings(lines: list[str]) -> tuple[list[str], int]:
    """Add ``# `` to lines such as ``Chapter 12`` on a line by themselves, and return how many."""

    def blank(index: int) -> bool:
        return index < 0 or index >= len(lines) or not lines[index].strip()

    found, index, result = 0, 0, []
    while index < len(lines):
        line = lines[index].strip()
        candidate = len(line) <= 80 and (CHAPTER_LINE.match(line) or ROMAN_LINE.match(line))
        if candidate and blank(index - 1) and blank(index + 1):
            heading = _title_case(line.rstrip(".:"))

            # Join a short title line that follows by itself, as "CHAPTER I." then "Down the Hole"
            after = index + 2
            title = lines[after].strip() if after < len(lines) and blank(index + 1) else ""
            if (
                title
                and len(title) <= 60
                and not ENDS_A_SENTENCE.search(title)
                and blank(after + 1)
                and not CHAPTER_LINE.match(title)
            ):
                heading = f"{heading}. {_title_case(title)}"
                index = after
            result += [f"# {heading}"]
            found += 1
        else:
            result.append(lines[index])
        index += 1
    return result, found


def _finish(title: str, text: str, warnings: Warnings) -> Book:
    """Build the result: measure the chapters and warn when the text is over the job limits."""
    chapters = [
        {"title": chapter.title, "chars": len(chapter.body)}
        for chapter in split_chapters(text, title or "Opening")
    ]
    if len(text) > MAX_JOB_CHARS:
        warnings.add(
            "over the job limit", f"{len(text):,} characters; the limit is {MAX_JOB_CHARS:,}"
        )
    if len(chapters) > MAX_CHAPTERS:
        warnings.add(
            "over the chapter limit", f"{len(chapters)} chapters; the limit is {MAX_CHAPTERS}"
        )
    return Book(title, text, chapters, warnings.as_list())


def import_text(raw: bytes, options: Options) -> Book:
    """Import a plain-text or Markdown file.

    Existing ``#`` and ``##`` headings are preserved. Without any, a line such as ``Chapter 12``
    or ``XII`` between blank lines becomes a heading.

    Parameters
    ----------
    raw : bytes
        File contents, in UTF-8 or Windows-1252.
    options : Options
        Which citations and links to remove.

    Returns
    -------
    Book
        Text in chapter format, with warnings.

    Raises
    ------
    ImportFailureError
        If the bytes are not text.
    """
    if b"\x00" in raw[:4096]:
        raise ImportFailureError("the file is not text")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    warnings = Warnings()
    text = HYPHEN_BREAK.sub(r"\1", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = text.split("\n")
    if not any(HEADING.match(line) for line in lines):
        lines, found = _detect_headings(lines)
        if found:
            warnings.add("chapter heading detected", "", found)

    # Clean each body line and drop lines that are only a page number
    cleaned = []
    for line in lines:
        if HEADING.match(line):
            cleaned.append(line)
        elif PAGE_NUMBER.match(line):
            warnings.add("page number removed", line)
        else:
            cleaned.append(clean_line(line, options, warnings))
    return _finish("", re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip() + "\n", warnings)


# ---- EPUB ----


class _Archive:
    """A zip file opened with limits on its entry count and sizes."""

    def __init__(self, data: bytes) -> None:
        try:
            self._zip = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise ImportFailureError("the file is not a valid EPUB (zip) archive") from exc
        infos = self._zip.infolist()
        if len(infos) > MAX_ENTRIES:
            raise ImportFailureError(f"the archive has more than {MAX_ENTRIES} entries")
        if sum(info.file_size for info in infos) > MAX_TOTAL_BYTES:
            raise ImportFailureError("the archive expands to more than 200 MB")
        self._infos = {info.filename: info for info in infos}

    def read(self, name: str) -> bytes | None:
        """Return an entry's bytes, or None when it is missing or over the size limit."""
        info = self._infos.get(name)
        if info is None or info.file_size > MAX_ENTRY_BYTES:
            return None
        return self._zip.read(info)


def _xml(data: bytes | None, what: str) -> ET.Element:
    """Parse XML from an EPUB's package files.

    The parser is the standard library's expat, which limits entity expansion and does not
    load external entities, so a hostile file cannot read other files or exhaust memory.
    """
    if data is None:
        raise ImportFailureError(f"the EPUB has no {what}")
    try:
        return ET.fromstring(data)  # noqa: S314 (expat limits expansion; see docstring)
    except ET.ParseError as exc:
        raise ImportFailureError(f"the EPUB's {what} is not valid XML") from exc


def _resolve(base: str, href: str) -> str:
    """Return the archive path of ``href`` written relative to the folder ``base``."""
    return posixpath.normpath(posixpath.join(base, unquote(href.split("#")[0])))


class _NavParser(HTMLParser):
    """Collect the links of an EPUB 3 navigation document's table of contents, in order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.entries: list[tuple[str, str]] = []
        self._in_toc = False
        self._href = ""
        self._label: list[str] = []
        self._in_link = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "nav" and "toc" in (a.get("epub:type") or "").split():
            self._in_toc = True
        elif tag == "a" and self._in_toc and a.get("href"):
            self._href, self._label, self._in_link = a["href"] or "", [], True

    def handle_endtag(self, tag: str) -> None:
        if tag == "nav":
            self._in_toc = False
        elif tag == "a" and self._in_link:
            self.entries.append((self._href, re.sub(r"\s+", " ", "".join(self._label)).strip()))
            self._in_link = False

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._label.append(data)


@dataclass
class _Package:
    """What the EPUB's package file says: title, spine files, and table of contents."""

    title: str
    spine: list[str]
    toc: list[tuple[str, str | None, str]]


def _read_package(archive: _Archive) -> _Package:
    """Read the package file and the table of contents it points to."""
    container = _xml(archive.read("META-INF/container.xml"), "META-INF/container.xml")
    rootfile = container.find(".//{*}rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        raise ImportFailureError("the EPUB's container.xml points to no package file")
    opf_path = posixpath.normpath(unquote(rootfile.get("full-path", "")))
    package = _xml(archive.read(opf_path), "package file")
    base = posixpath.dirname(opf_path)

    manifest: dict[str, tuple[str, str, set[str]]] = {}
    for item in package.iterfind("{*}manifest/{*}item"):
        manifest[item.get("id", "")] = (
            _resolve(base, item.get("href", "")),
            item.get("media-type", ""),
            set((item.get("properties") or "").split()),
        )

    # Take the spine files in reading order, without the navigation document or non-linear items
    spine_el = package.find("{*}spine")
    spine = []
    if spine_el is not None:
        for ref in spine_el.iterfind("{*}itemref"):
            path, _, props = manifest.get(ref.get("idref", ""), ("", "", set()))
            if path and ref.get("linear") != "no" and "nav" not in props:
                spine.append(path)

    # Read the table of contents: the EPUB 3 navigation document, else the NCX
    toc: list[tuple[str, str | None, str]] = []
    nav = next((path for path, _, props in manifest.values() if "nav" in props), None)
    ncx_id = spine_el.get("toc", "") if spine_el is not None else ""
    ncx = manifest[ncx_id][0] if ncx_id in manifest else None
    raw_nav = archive.read(nav) if nav else None
    if raw_nav is not None:
        parser = _NavParser()
        parser.feed(raw_nav.decode("utf-8", errors="replace"))
        entries = [(href, label) for href, label in parser.entries if label]
        toc_base = posixpath.dirname(nav or "")
    elif ncx:
        root = _xml(archive.read(ncx), "table of contents")
        entries = []
        for point in root.iter("{*}navPoint"):
            label = re.sub(r"\s+", " ", point.findtext("{*}navLabel/{*}text") or "").strip()
            content = point.find("{*}content")
            if label and content is not None:
                entries.append((content.get("src", ""), label))
        toc_base = posixpath.dirname(ncx)
    else:
        entries, toc_base = [], base
    for href, label in entries:
        fragment = href.split("#", 1)[1] if "#" in href else None
        toc.append((_resolve(toc_base, href), fragment, label))
    return _Package(
        re.sub(r"\s+", " ", package.findtext("{*}metadata/{*}title") or "").strip(), spine, toc
    )


class _Body(HTMLParser):
    """Turn one XHTML file into paragraph blocks, with markers at the table of contents anchors.

    Parameters
    ----------
    targets : set of str
        Element ids where a chapter may start; a ``mark`` token is emitted before each.
    options : Options
        Whether figure captions are read.
    warnings : Warnings
        Receives a count for each kind of content left out.
    """

    def __init__(self, targets: set[str], options: Options, warnings: Warnings) -> None:
        super().__init__(convert_charrefs=True)
        self.targets, self.options, self.warnings = targets, options, warnings
        self.tokens: list[tuple[str, str]] = []
        self.types: set[str] = set()
        self.first_heading = ""
        self._stack: list[tuple[str, bool]] = []
        self._skipping = 0
        self._parts: list[str] = []
        self._block = "p"
        self._inline: list[tuple[str, int, bool, bool]] = []

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self._parts)).strip()
        self._parts.clear()
        if not text:
            self._block = "p"
            return
        if self._block in HEADING_TAGS and not self.first_heading:
            self.first_heading = text

        # Close headings and list items with a period so combined chunks do not run together
        if (self._block in HEADING_TAGS or self._block == "li") and not ENDS_A_SENTENCE.search(
            text
        ):
            text += "."
        self.tokens.append(("block", text))
        self._block = "p"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        kinds = set((a.get("epub:type") or a.get("role") or "").replace("doc-", "").split())
        direct_child = bool(self._stack) and self._stack[-1][0] == "body"
        if tag == "body" or direct_child:
            self.types |= kinds
        if tag in VOID_TAGS:
            if tag == "br" and not self._skipping:
                self._parts.append(" ")
            return

        # Decide whether this element's content is left out
        skip = tag in SKIP_TAGS or bool(kinds & NOTE_TYPES)
        if tag == "table" and not self._skipping:
            self.warnings.add("table skipped")
        if tag == "figure" and not self.options.captions:
            skip = True
            if not self._skipping:
                self.warnings.add("figure skipped")
        if tag == "math" and not self._skipping:
            alt = a.get("alttext", "").strip()
            if alt:
                self._parts.append(f" {alt} ")
            self.warnings.add("equation", alt)
            skip = True
        if kinds & NOTE_TYPES and not self._skipping:
            self.warnings.add("footnote skipped")
        self._stack.append((tag, skip))
        if skip:
            self._skipping += 1
        if self._skipping:
            return

        # Start a new block, and mark an anchor where a chapter may begin
        if tag in BLOCK_TAGS:
            self._flush()
            self._block = tag
        if a.get("id") in self.targets:
            self._flush()
            self.tokens.append(("mark", a["id"]))

        # Remember where a link or superscript starts, to remove it if it is a note marker
        if tag in ("a", "sup"):
            linked = "#" in a.get("href", "")
            self._inline.append((tag, len(self._parts), "noteref" in kinds, linked))

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        position = next(
            (i for i in range(len(self._stack) - 1, -1, -1) if self._stack[i][0] == tag), None
        )
        if position is None:
            return

        # Close any elements left open inside this one, then this element
        while len(self._stack) > position:
            closing, skipped = self._stack.pop()
            if skipped:
                self._skipping -= 1
                continue
            if closing in ("a", "sup") and self._inline and self._inline[-1][0] == closing:
                _, start, noteref, linked = self._inline.pop()
                text = "".join(self._parts[start:]).strip()
                if closing == "a":
                    marker = bool(text) and (noteref or (linked and bool(NOTE_MARKER.match(text))))
                else:
                    marker = bool(re.fullmatch(r"[*\u2020\u2021\u00a7]+", text))
                if marker:
                    del self._parts[start:]
                    self.warnings.add("note marker removed", text)
            if closing in BLOCK_TAGS:
                self._flush()

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self._parts.append(data)

    def close(self) -> None:
        """Finish parsing and write out the last block."""
        super().close()
        self._flush()


def _clean_title(label: str) -> str:
    """Return a chapter title on one line, without ``#`` and at most ``MAX_TITLE_CHARS`` long."""
    return re.sub(r"\s+", " ", label.replace("#", " ")).strip()[:MAX_TITLE_CHARS]


def _same_words(a: str, b: str) -> bool:
    """Return whether two strings have the same letters and digits, ignoring case and marks."""
    squash = re.compile(r"\W+")
    return squash.sub("", a).lower() == squash.sub("", b).lower()


def import_epub(raw: bytes, options: Options) -> Book:
    """Import an EPUB: chapters from its table of contents, body paragraphs, and a cleanup pass.

    A chapter starts at each table of contents entry, at its anchor inside a file or at the
    start of the file, so several chapters in one file and one chapter over several files
    work. Text before the first entry (cover, title page, copyright) is left out.
    Without a table of contents, each file is one chapter.

    Parameters
    ----------
    raw : bytes
        The EPUB file.
    options : Options
        Which citations, links, and figure captions to remove.

    Returns
    -------
    Book
        Text in chapter format, with warnings that count what was left out.

    Raises
    ------
    ImportFailureError
        If the file is not an EPUB, is missing its package file, or exceeds the size limits.
    """
    archive = _Archive(raw)
    package = _read_package(archive)
    warnings = Warnings()
    if not package.spine:
        raise ImportFailureError("the EPUB has no readable content files")
    wanted: dict[str, dict[str | None, str]] = {}
    for path, fragment, label in package.toc:
        wanted.setdefault(path, {}).setdefault(fragment, label)

    # Parse each content file in reading order into tokens: a chapter start or a paragraph
    stream: list[tuple[str, str]] = []
    fallback = not package.toc
    for number, path in enumerate(package.spine, start=1):
        data = archive.read(path)
        if data is None:
            warnings.add("missing file", path)
            continue
        labels = wanted.get(path, {})
        parser = _Body({fragment for fragment in labels if fragment}, options, warnings)
        parser.feed(data.decode("utf-8", errors="replace"))
        parser.close()
        if parser.types & SKIP_TYPES:
            warnings.add("section skipped", sorted(parser.types & SKIP_TYPES)[0])
            continue
        found = {value for kind, value in parser.tokens if kind == "mark"}

        # Start a chapter at the top of the file for an entry with no anchor or a missing anchor
        if fallback:
            stream.append(("start", parser.first_heading or f"Section {number}"))
        else:
            top = labels.get(None) or next(
                (v for k, v in labels.items() if k and k not in found), ""
            )
            if top:
                stream.append(("start", top))
        for kind, value in parser.tokens:
            stream.append(("start", labels[value]) if kind == "mark" else (kind, value))
    if fallback:
        warnings.add("no table of contents", "one chapter per file")

    # Group the paragraphs under their chapter starts, dropping text that comes before the first
    chapters: list[tuple[str, list[str]]] = []
    skipped = 0
    for kind, value in stream:
        if kind == "start":
            chapters.append((_clean_title(value), []))
        elif chapters:
            chapters[-1][1].append(value)
        else:
            skipped += len(value)
    if skipped:
        warnings.add("text before the first chapter skipped", "", skipped)

    # Clean the paragraphs of each chapter, and drop a first line that repeats the title
    sections = []
    for title, blocks in chapters:
        paragraphs: list[str] = []
        for number, block in enumerate(blocks):
            if number == 0 and _same_words(block, title):
                continue
            block = re.sub(r"^#{1,2}[ \t]+", "", clean_line(block, options, warnings))
            if block and not PAGE_NUMBER.match(block):
                paragraphs.append(block)
        if paragraphs and title:
            sections.append(f"# {title}\n\n" + "\n\n".join(paragraphs))
    return _finish(package.title, "\n\n".join(sections) + "\n", warnings)


def import_document(raw: bytes, filename: str, options: Options) -> Book:
    """Import an uploaded file by its type: EPUB, plain text, or Markdown.

    Parameters
    ----------
    raw : bytes
        The file contents.
    filename : str
        The uploaded file name, used with the contents to tell the type.
    options : Options
        Which citations, links, and figure captions to remove.

    Returns
    -------
    Book
        Text in chapter format, with warnings.

    Raises
    ------
    ImportFailureError
        If the type is not supported or the file is damaged.
    """
    if raw[:5] == b"%PDF-" or filename.lower().endswith(".pdf"):
        raise ImportFailureError("PDF import is not supported yet; paste the text or use an EPUB")
    if raw[:4] == b"PK\x03\x04" or filename.lower().endswith(".epub"):
        return import_epub(raw, options)
    return import_text(raw, options)
