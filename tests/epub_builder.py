"""Build small EPUB files in memory for the importer and supervisor tests."""

import io
import zipfile

XHTML = (
    '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml" '
    'xmlns:epub="http://www.idpf.org/2007/ops"><head><title>x</title></head>'
    "<body{attrs}>{body}</body></html>"
)


def page(body: str, attrs: str = "") -> str:
    """Return an XHTML document around ``body``, with optional attributes on the body tag."""
    return XHTML.format(body=body, attrs=attrs)


def make_epub(
    files: dict[str, str],
    toc: list[tuple[str, str]],
    *,
    kind: str = "nav",
    title: str = "A Book",
    nonlinear: tuple[str, ...] = (),
) -> bytes:
    """Build an EPUB in memory.

    Parameters
    ----------
    files : dict
        Content file path (inside ``OEBPS``) mapped to its XHTML, in reading order.
    toc : list of tuple
        Each table of contents entry as ``(href, label)``, with an optional ``#fragment``.
    kind : str
        ``"nav"`` for an EPUB 3 navigation document, ``"ncx"`` for EPUB 2, or ``"none"``.
    title : str
        The book title in the package file.
    nonlinear : tuple of str
        Files marked ``linear="no"`` in the spine.
    """
    items = "".join(
        f'<item id="f{i}" href="{n}" media-type="application/xhtml+xml"/>'
        for i, n in enumerate(files)
    )
    refs = "".join(
        f'<itemref idref="f{i}"{" linear=" + chr(34) + "no" + chr(34) if n in nonlinear else ""}/>'
        for i, n in enumerate(files)
    )
    spine_attr = ' toc="ncx"' if kind == "ncx" else ""
    if kind == "nav":
        items += (
            '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
        )
    if kind == "ncx":
        items += '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
    opf = (
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata '
        f'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title></metadata>'
        f"<manifest>{items}</manifest><spine{spine_attr}>{refs}</spine></package>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        archive.writestr("OEBPS/content.opf", opf)
        for name, content in files.items():
            archive.writestr(f"OEBPS/{name}", content)
        if kind == "nav":
            links = "".join(f'<li><a href="{h}">{label}</a></li>' for h, label in toc)
            archive.writestr(
                "OEBPS/nav.xhtml",
                page(f'<nav epub:type="toc"><ol>{links}</ol></nav>'),
            )
        if kind == "ncx":
            points = "".join(
                f'<navPoint id="p{i}"><navLabel><text>{label}</text></navLabel>'
                f'<content src="{h}"/></navPoint>'
                for i, (h, label) in enumerate(toc)
            )
            archive.writestr(
                "OEBPS/toc.ncx",
                f'<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>{points}</navMap></ncx>',
            )
    return buffer.getvalue()
