"""
Pull books from Project Gutenberg via Gutendex (https://gutendex.com).

Gutendex is a free public API that wraps the Project Gutenberg catalog.
No API key required.

Setup:
    pip install requests

Usage:
    # Search for books
    python gutenberg.py search "shakespeare" --page-size 10

    # Download one book's plain text by ID into ../data/
    python gutenberg.py get 1513

    # Search and download the top N matches in one shot
    python gutenberg.py pull "sherlock holmes" --count 3

    # Concatenate every downloaded work in an author folder into a single
    # training file, with a separator between works so the model can learn
    # where one ends and the next begins.
    python gutenberg.py build --from poe --output input_poe.txt

    # Combine multiple folders, drop one book, and add an extra one by ID.
    python gutenberg.py build --from doyle --from twain --exclude 3178 74 --output input_combo.txt
"""

import argparse
import os
import re
import sys

import requests

BASE_URL = "https://gutendex.com"

# Save into the `data` folder that sits next to this `code` folder.
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# Preferred plain-text format keys, in order.
TEXT_FORMATS = ["text/plain; charset=utf-8", "text/plain; charset=us-ascii", "text/plain"]

WORK_SEPARATOR = "\n\n<|endofwork|>\n\n"

# Every Gutenberg plaintext file wraps the actual book in a license
# header/footer (incl. the Project Gutenberg Literary Archive Foundation
# 501(c)(3) boilerplate). These START/END marker lines are a stable part of
# their distribution format across editions ("THIS"/"THE" wording varies by
# decade), so we can reliably cut everything outside them.
GUTENBERG_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
GUTENBERG_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)


def strip_boilerplate(text):
    """Drop Project Gutenberg's license header/footer, keeping only the
    actual book content between the START/END markers."""
    start_match = GUTENBERG_START_RE.search(text)
    if start_match:
        text = text[start_match.end():]
    end_match = GUTENBERG_END_RE.search(text)
    if end_match:
        text = text[:end_match.start()]
    return text.strip()


def _get(path, params=None):
    resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _slugify(text, max_len=60):
    slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug[:max_len] or "book"


def search(query, page_size=10, page=1):
    """Search for books; returns the list of result dicts."""
    data = _get("/books/", {"search": query, "page": page})
    results = data.get("results", [])
    return results[:page_size]


def get_book_meta(book_id):
    """Fetch metadata (including format URLs) for a single book ID."""
    return _get(f"/books/{book_id}")


def get_text(book_id):
    """Download the plain text of a book by ID, following Gutendex's format links."""
    meta = get_book_meta(book_id)
    formats = meta.get("formats", {})
    text_url = next((formats[k] for k in TEXT_FORMATS if k in formats), None)
    if not text_url:
        return "", meta
    resp = requests.get(text_url, timeout=60)
    resp.raise_for_status()
    return resp.text, meta


def save_book(book_id, subdir=None):
    """Download a book's text and write it into ../data/[subdir]/. Returns the file path."""
    target_dir = os.path.join(DATA_DIR, subdir) if subdir else DATA_DIR
    os.makedirs(target_dir, exist_ok=True)
    text, meta = get_text(book_id)
    if not text.strip():
        print(f"  ! book {book_id} has no plain-text format available, skipping")
        return None
    text = strip_boilerplate(text)
    title = meta.get("title", "")
    name = f"{book_id}_{_slugify(title)}.txt" if title else f"{book_id}.txt"
    path = os.path.join(target_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  saved {len(text):,} chars -> {path}")
    return path


def _print_results(results):
    for b in results:
        bid = b.get("id", "?")
        title = b.get("title", "<untitled>")
        authors = ", ".join(a.get("name", "") for a in b.get("authors", []))
        print(f"  [{bid}] {title}" + (f" — {authors}" if authors else ""))


def cmd_search(args):
    results = search(args.query, args.page_size, args.page)
    print(f"Found {len(results)} result(s) for '{args.query}':")
    _print_results(results)


def cmd_get(args):
    print(f"Downloading book {args.id}...")
    save_book(args.id, subdir=args.dir)


def cmd_pull(args):
    results = search(args.query, page_size=args.count)
    print(f"Pulling top {len(results)} match(es) for '{args.query}':")
    for b in results[: args.count]:
        bid = b.get("id")
        if bid is None:
            continue
        print(f"  [{bid}] {b.get('title', '')}")
        save_book(bid, subdir=args.dir)


def _iter_dir_book_files(dirpath):
    """Yield (book_id, filename) for every downloaded book in dirpath, sorted by id."""
    entries = []
    for fname in os.listdir(dirpath):
        m = re.match(r"^(\d+)(?:_.*)?\.txt$", fname)
        if m:
            entries.append((int(m.group(1)), fname))
    return sorted(entries)


def _find_book_file(book_id):
    """Search ../data/ (including author subfolders) for a downloaded file matching this ID."""
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if fname.startswith(f"{book_id}_") or fname == f"{book_id}.txt":
                return os.path.join(root, fname)
    return None


def build_input(output_path, from_dirs=None, include_ids=None, exclude_ids=None):
    """Concatenate downloaded works into one training file.

    from_dirs: author subfolder names under ../data/ to include in full.
    include_ids: specific book IDs to also include (searched across all of ../data/).
    exclude_ids: book IDs to skip, even if picked up via from_dirs.
    """
    exclude_ids = set(exclude_ids or [])
    chunks = []

    for dirname in from_dirs or []:
        dirpath = os.path.join(DATA_DIR, dirname)
        if not os.path.isdir(dirpath):
            print(f"  ! no such directory: {dirpath}")
            continue
        for bid, fname in _iter_dir_book_files(dirpath):
            if bid in exclude_ids:
                print(f"  - {fname} (excluded)")
                continue
            with open(os.path.join(dirpath, fname), "r", encoding="utf-8") as f:
                chunks.append(f.read())
            print(f"  + {fname}")

    for bid in include_ids or []:
        if bid in exclude_ids:
            continue
        path = _find_book_file(bid)
        if not path:
            print(f"  ! no downloaded file found for book {bid}, skipping (run 'get {bid}' first)")
            continue
        with open(path, "r", encoding="utf-8") as f:
            chunks.append(f.read())
        print(f"  + {os.path.basename(path)}")

    if not chunks:
        sys.exit("No works found to build from.")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(WORK_SEPARATOR.join(chunks))
    total_chars = sum(len(c) for c in chunks)
    print(f"Wrote {total_chars:,} chars from {len(chunks)} work(s) -> {output_path}")


def cmd_build(args):
    build_input(args.output, from_dirs=args.from_dirs, include_ids=args.ids, exclude_ids=args.exclude)


def main():
    parser = argparse.ArgumentParser(description="Pull Project Gutenberg books via Gutendex.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="search for books")
    p.add_argument("query")
    p.add_argument("--page-size", type=int, default=10)
    p.add_argument("--page", type=int, default=1)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("get", help="download one book by ID into ../data/[--dir]/")
    p.add_argument("id", type=int)
    p.add_argument("--dir", default=None, help="author subfolder under ../data/, e.g. 'twain'")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("pull", help="search then download the top matches")
    p.add_argument("query")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--dir", default=None, help="author subfolder under ../data/, e.g. 'twain'")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("build", help="build a training file from author folders and/or specific book IDs")
    p.add_argument("ids", type=int, nargs="*",
                    help="specific book IDs to also include (searched across all of ../data/)")
    p.add_argument("--from", dest="from_dirs", action="append", default=[], metavar="DIR",
                    help="author subfolder under ../data/ to include in full (repeatable)")
    p.add_argument("--exclude", type=int, nargs="+", default=[],
                    help="book IDs to skip, even if picked up via --from")
    p.add_argument("--output", default="input.txt")
    p.set_defaults(func=cmd_build)

    args = parser.parse_args()
    try:
        args.func(args)
    except requests.HTTPError as e:
        sys.exit(f"HTTP error: {e}")


if __name__ == "__main__":
    main()
