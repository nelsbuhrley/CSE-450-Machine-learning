# CSE-450-Machine-learning

My work for the CSE 450 class.

## Tools

This repository includes three command-line tools:

1. `tools/extractCSV.py` downloads a CSV from a URL and saves it locally with the same filename.
2. `tools/compareCSV.py` compares two CSV files, reports how similar they are, and can write derived CSV outputs.
3. `tools/notebook_to_py.py` converts a Jupyter notebook into a plain Python file.

Run the commands from the repository root.

## `tools/extractCSV.py`

Downloads a web-based CSV file and saves it to a local directory using the same filename as the source URL.

### Usage

```bash
python tools/extractCSV.py <csv-url> [-d DESTINATION]
```

### Arguments

`<csv-url>`
: HTTP or HTTPS URL to the CSV file.

`-d`, `--destination`
: Directory where the file should be saved. The directory is created if it does not already exist. Defaults to the current directory.

### Examples

Save a file in the current directory:

```bash
python tools/extractCSV.py https://example.com/data/sample.csv
```

Save a file into a separate folder:

```bash
python tools/extractCSV.py https://example.com/data/sample.csv --destination downloads
```

### Behavior

- The output filename is taken from the last path segment of the URL.
- The file is written exactly once with the same name as the remote file.
- If the URL path does not contain a filename, the script raises an error.

## `tools/compareCSV.py`

Compares two CSV files and prints how much data they share in both directions. It can also write derived CSVs for the sum, differences, intersection, and XOR of the rows.

The script is tolerant of small differences:

- headers are compared case-insensitively after trimming whitespace
- text values are compared case-insensitively unless you opt out
- numeric values can be compared with a tolerance
- rows are matched by normalized values instead of raw text
- duplicate rows are handled correctly

### Usage

```bash
python tools/compareCSV.py <csv-a> <csv-b> [options]
```

### Arguments

`<csv-a>`
: First CSV file.

`<csv-b>`
: Second CSV file.

### Options

`--sum-out PATH`
: Write the row-wise sum of both files to `PATH`.

`--diff12-out PATH`
: Write rows that appear in the first file but not the second.

`--diff21-out PATH`
: Write rows that appear in the second file but not the first.

`--and-out PATH`
: Write rows shared by both files.

`--xor-out PATH`
: Write rows that appear in only one of the files.

`--numeric-tolerance FLOAT`
: Treat numeric values within this absolute tolerance as equal. Default is `0`, which means exact numeric comparison.

`--case-sensitive`
: Compare text values with case sensitivity. By default, text comparison ignores case.

`--output-delimiter DELIMITER`
: Use this delimiter when writing generated output files. If omitted, the delimiter from the first CSV is used.

### Output report

The script always prints a two-line similarity summary unless it cannot compare the files.

Example output:

```text
file1.csv shares 100.00% of its data with file2.csv (25/25 rows)
file2.csv shares 96.15% of its data with file1.csv (25/26 rows)
```

If the files do not use identical headers, the script also prints a header-alignment note showing how many shared columns were compared.

### Examples

Compare two CSV files:

```bash
python tools/compareCSV.py data/a.csv data/b.csv
```

Compare two files with numeric tolerance and write the rows common to both:

```bash
python tools/compareCSV.py data/a.csv data/b.csv --numeric-tolerance 0.1 --and-out output/common.csv
```

Write every derived output in one run:

```bash
python tools/compareCSV.py data/a.csv data/b.csv \
	--sum-out output/sum.csv \
	--diff12-out output/a_minus_b.csv \
	--diff21-out output/b_minus_a.csv \
	--and-out output/intersection.csv \
	--xor-out output/xor.csv
```

### Behavior

- Rows are matched using normalized header names, so `Name` and `name` are treated as the same column.
- Numeric values such as `1`, `1.0`, and `1.00` are treated as equivalent unless you set a tighter tolerance.
- The generated CSVs keep the column order from the two input files, combined without duplicates.
- If you request no output files, the script only prints the similarity summary.

## `notebook_to_py.py`

Converts a Jupyter notebook into a `.py` file. Markdown cells become comment blocks and code cells stay as code.

### Usage

```bash
python notebook_to_py.py <notebook.ipynb> [output.py]
```

### Arguments

`<notebook.ipynb>`
: Input notebook file.

`[output.py]`
: Optional destination Python file. If omitted, the script writes a `.py` file next to the notebook with the same base name.

### Examples

Convert a notebook in place to a Python file with the same base name:

```bash
python notebook_to_py.py notebooks/Exploration_01.ipynb
```

Convert a notebook and choose a custom output path:

```bash
python notebook_to_py.py notebooks/Exploration_01.ipynb notebooks/Exploration_01_export.py
```

### Behavior

- The input file must end in `.ipynb`.
- The script fails if the notebook does not exist.
- Markdown text is converted into `#` comment lines in the output file.
- Empty notebook cells are skipped.
