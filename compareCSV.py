#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
	script_path = Path(__file__).resolve().parent / "Tools" / "compareCSV.py"
	runpy.run_path(str(script_path), run_name="__main__")"""Compare two CSV files and optionally write row-wise set-operation outputs.

The script is designed to be forgiving about small format differences:
- headers are matched case-insensitively after stripping whitespace
- numeric cells can be compared with an optional tolerance
- rows are matched on normalized keys, not raw text
- duplicate rows are handled with multiset semantics
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Sequence


DELIMITERS = [",", ";", "\t", "|"]
NUMERIC_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass(slots=True)
class CsvTable:
	path: Path
	headers: list[str]
	rows: list[dict[str, str]]
	delimiter: str


def normalize_header(header: str) -> str:
	return re.sub(r"\s+", "_", header.strip().lower())


def normalize_text(value: str, ignore_case: bool) -> str:
	value = value.strip()
	return value.lower() if ignore_case else value


def format_number(value: Decimal, tolerance: Decimal | None) -> str:
	if tolerance and tolerance > 0:
		rounded = (value / tolerance).to_integral_value(rounding="ROUND_HALF_UP") * tolerance
		value = rounded
	normalized = format(value.normalize(), "f")
	if normalized in {"-0", "-0.0"}:
		return "0"
	return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def normalize_value(value: str | None, *, ignore_case: bool, numeric_tolerance: Decimal | None) -> str:
	if value is None:
		return ""
	text = value.strip()
	if text == "":
		return ""
	if NUMERIC_PATTERN.match(text):
		try:
			number = Decimal(text)
		except InvalidOperation:
			pass
		else:
			return format_number(number, numeric_tolerance)
	return normalize_text(text, ignore_case)


def detect_dialect(path: Path):
	sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
	sniffer = csv.Sniffer()
	try:
		return sniffer.sniff(sample, delimiters=",".join(DELIMITERS))
	except csv.Error:
		dialect = csv.get_dialect("excel")
		return dialect


def read_csv_table(path: Path) -> CsvTable:
	dialect = detect_dialect(path)
	with path.open("r", encoding="utf-8-sig", newline="") as handle:
		reader = csv.DictReader(handle, dialect=dialect)
		headers = list(reader.fieldnames or [])
		rows: list[dict[str, str]] = []
		for raw_row in reader:
			if raw_row is None:
				continue
			if not any((value or "").strip() for value in raw_row.values()):
				continue
			rows.append({key: (value if value is not None else "") for key, value in raw_row.items()})
	if not headers:
		raise ValueError(f"{path} does not appear to contain a CSV header row.")
	return CsvTable(path=path, headers=headers, rows=rows, delimiter=getattr(dialect, "delimiter", ","))


def build_header_maps(headers_a: Sequence[str], headers_b: Sequence[str]) -> tuple[list[str], dict[str, str], dict[str, str]]:
	map_a = {normalize_header(header): header for header in headers_a}
	map_b = {normalize_header(header): header for header in headers_b}
	common = set(map_a) & set(map_b)
	if not common:
		raise ValueError("The CSV files do not share any comparable headers.")
	compare_headers = [normalize_header(name) for name in headers_a if normalize_header(name) in common]
	return compare_headers, map_a, map_b


def choose_output_headers(table_a: CsvTable, table_b: CsvTable) -> list[str]:
	seen: set[str] = set()
	output_headers: list[str] = []
	for header in list(table_a.headers) + list(table_b.headers):
		normalized = normalize_header(header)
		if normalized in seen:
			continue
		seen.add(normalized)
		output_headers.append(header)
	return output_headers


def row_key(row: dict[str, str], compare_headers: Sequence[str], *, ignore_case: bool, numeric_tolerance: Decimal | None) -> tuple[str, ...]:
	normalized_row = {normalize_header(key): value for key, value in row.items() if key is not None}
	return tuple(
		normalize_value(normalized_row.get(header), ignore_case=ignore_case, numeric_tolerance=numeric_tolerance)
		for header in compare_headers
	)


def build_multiset_index(
	table: CsvTable,
	compare_headers: Sequence[str],
	*,
	ignore_case: bool,
	numeric_tolerance: Decimal | None,
) -> tuple[list[tuple[str, ...]], dict[tuple[str, ...], list[dict[str, str]]], Counter]:
	keys: list[tuple[str, ...]] = []
	grouped_rows: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
	counts: Counter = Counter()
	for row in table.rows:
		key = row_key(row, compare_headers, ignore_case=ignore_case, numeric_tolerance=numeric_tolerance)
		keys.append(key)
		grouped_rows[key].append(row)
		counts[key] += 1
	return keys, grouped_rows, counts


def count_matches(counts_a: Counter, counts_b: Counter) -> int:
	return sum(min(counts_a[key], counts_b[key]) for key in counts_a.keys() & counts_b.keys())


def build_union_row(row: dict[str, str], headers: Sequence[str]) -> dict[str, str]:
	normalized_row = {normalize_header(key): value for key, value in row.items() if key is not None}
	return {header: normalized_row.get(normalize_header(header), "") for header in headers}


def select_rows_for_operation(
	table_a: CsvTable,
	table_b: CsvTable,
	compare_headers: Sequence[str],
	*,
	ignore_case: bool,
	numeric_tolerance: Decimal | None,
) -> dict[str, list[dict[str, str]]]:
	keys_a, grouped_a, counts_a = build_multiset_index(
		table_a,
		compare_headers,
		ignore_case=ignore_case,
		numeric_tolerance=numeric_tolerance,
	)
	keys_b, grouped_b, counts_b = build_multiset_index(
		table_b,
		compare_headers,
		ignore_case=ignore_case,
		numeric_tolerance=numeric_tolerance,
	)

	matched_counts = Counter({key: min(counts_a[key], counts_b[key]) for key in counts_a.keys() & counts_b.keys()})
	seen_a = Counter()
	seen_b = Counter()
	result_and: list[dict[str, str]] = []
	result_diff_a: list[dict[str, str]] = []
	result_diff_b: list[dict[str, str]] = []

	for key in keys_a:
		seen_a[key] += 1
		row = grouped_a[key][seen_a[key] - 1]
		if seen_a[key] <= matched_counts[key]:
			result_and.append(row)
		else:
			result_diff_a.append(row)

	for key in keys_b:
		seen_b[key] += 1
		if seen_b[key] > matched_counts[key]:
			result_diff_b.append(grouped_b[key][seen_b[key] - 1])

	return {
		"and": result_and,
		"diff_a": result_diff_a,
		"diff_b": result_diff_b,
		"xor": result_diff_a + result_diff_b,
	}


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[dict[str, str]], delimiter: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(headers), delimiter=delimiter)
		writer.writeheader()
		for row in rows:
			writer.writerow(build_union_row(row, headers))


def compute_similarity(table_a: CsvTable, table_b: CsvTable, compare_headers: Sequence[str], *, ignore_case: bool, numeric_tolerance: Decimal | None) -> tuple[int, int, int, float, float]:
	_, _, counts_a = build_multiset_index(
		table_a,
		compare_headers,
		ignore_case=ignore_case,
		numeric_tolerance=numeric_tolerance,
	)
	_, _, counts_b = build_multiset_index(
		table_b,
		compare_headers,
		ignore_case=ignore_case,
		numeric_tolerance=numeric_tolerance,
	)
	matched = count_matches(counts_a, counts_b)
	total_a = len(table_a.rows)
	total_b = len(table_b.rows)
	pct_a = (matched / total_a * 100.0) if total_a else 0.0
	pct_b = (matched / total_b * 100.0) if total_b else 0.0
	return matched, total_a, total_b, pct_a, pct_b


def describe_similarity(table_a: CsvTable, table_b: CsvTable, compare_headers: Sequence[str], *, ignore_case: bool, numeric_tolerance: Decimal | None) -> str:
	matched, total_a, total_b, pct_a, pct_b = compute_similarity(
		table_a,
		table_b,
		compare_headers,
		ignore_case=ignore_case,
		numeric_tolerance=numeric_tolerance,
	)
	name_a = table_a.path.name
	name_b = table_b.path.name
	lines = [
		f"{name_a} shares {pct_a:.2f}% of its data with {name_b} ({matched}/{total_a} rows)",
		f"{name_b} shares {pct_b:.2f}% of its data with {name_a} ({matched}/{total_b} rows)",
	]
	if table_a.headers != table_b.headers:
		lines.append(
			f"Header alignment: comparing {len(compare_headers)} shared columns across {len(table_a.headers)} and {len(table_b.headers)} columns"
		)
	return "\n".join(lines)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compare two CSV files, report how similar they are, and optionally write row-wise set-operation outputs."
	)
	parser.add_argument("csv_a", type=Path, help="First CSV file")
	parser.add_argument("csv_b", type=Path, help="Second CSV file")
	parser.add_argument("--sum-out", type=Path, help="Write the row-wise sum/union of both files")
	parser.add_argument("--diff12-out", type=Path, help="Write rows in the first file that are not in the second")
	parser.add_argument("--diff21-out", type=Path, help="Write rows in the second file that are not in the first")
	parser.add_argument("--and-out", type=Path, help="Write rows common to both files")
	parser.add_argument("--xor-out", type=Path, help="Write rows that appear in only one file")
	parser.add_argument(
		"--numeric-tolerance",
		type=float,
		default=0.0,
		help="Treat numeric cells within this absolute tolerance as equal (default: 0)",
	)
	parser.add_argument(
		"--case-sensitive",
		action="store_true",
		help="Compare text values with case sensitivity instead of normalizing case",
	)
	parser.add_argument(
		"--output-delimiter",
		default=None,
		help="Delimiter to use for generated output files. Defaults to the first file's delimiter.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	table_a = read_csv_table(args.csv_a)
	table_b = read_csv_table(args.csv_b)
	compare_headers, _, _ = build_header_maps(table_a.headers, table_b.headers)

	ignore_case = not args.case_sensitive
	numeric_tolerance = Decimal(str(args.numeric_tolerance)) if args.numeric_tolerance else None
	output_delimiter = args.output_delimiter or table_a.delimiter or ","

	print(
		describe_similarity(
			table_a,
			table_b,
			compare_headers,
			ignore_case=ignore_case,
			numeric_tolerance=numeric_tolerance,
		)
	)

	requested_outputs = any(
		value is not None
		for value in (args.sum_out, args.diff12_out, args.diff21_out, args.and_out, args.xor_out)
	)
	if not requested_outputs:
		return 0

	outputs = select_rows_for_operation(
		table_a,
		table_b,
		compare_headers,
		ignore_case=ignore_case,
		numeric_tolerance=numeric_tolerance,
	)
	output_headers = choose_output_headers(table_a, table_b)

	if args.sum_out:
		write_csv(args.sum_out, output_headers, [*table_a.rows, *table_b.rows], output_delimiter)
	if args.diff12_out:
		write_csv(args.diff12_out, output_headers, outputs["diff_a"], output_delimiter)
	if args.diff21_out:
		write_csv(args.diff21_out, output_headers, outputs["diff_b"], output_delimiter)
	if args.and_out:
		write_csv(args.and_out, output_headers, outputs["and"], output_delimiter)
	if args.xor_out:
		write_csv(args.xor_out, output_headers, outputs["xor"], output_delimiter)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
