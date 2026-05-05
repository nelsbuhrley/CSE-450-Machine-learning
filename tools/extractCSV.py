"""Download a CSV from a web URL and save it locally with the same filename."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse, unquote
from urllib.request import urlopen


def infer_filename(source_url: str) -> str:
	"""Return the filename portion of a URL path."""

	path = unquote(urlparse(source_url).path)
	filename = Path(path).name
	if not filename:
		raise ValueError("The URL must include a CSV filename in its path.")
	return filename


def download_csv(source_url: str, destination_dir: str | Path = ".") -> Path:
	"""Download a CSV from `source_url` into `destination_dir`."""

	destination_path = Path(destination_dir)
	destination_path.mkdir(parents=True, exist_ok=True)

	filename = infer_filename(source_url)
	output_path = destination_path / filename

	with urlopen(source_url) as response:
		content = response.read()

	output_path.write_bytes(content)
	return output_path


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Download a web-based CSV file and save it locally with the same name."
	)
	parser.add_argument("url", help="HTTP or HTTPS URL to the CSV file")
	parser.add_argument(
		"-d",
		"--destination",
		default=".",
		help="Directory to save the file in (defaults to the current directory)",
	)
	return parser


def main() -> int:
	args = build_parser().parse_args()
	output_path = download_csv(args.url, args.destination)
	print(output_path)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
